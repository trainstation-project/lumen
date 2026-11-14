"""Compiles lumen's matmuls on CUDA through the Ampere GEMM (`ampere.AmpereGemm`) for Rust to launch, the way QuACK
compiles its GEMMs: once per dtypes, majors, batching and tile, against fake tensors whose extents and strides are
symbolic (only each operand's contiguous dimension static, its other strides 16-byte multiples): A (m, k), B (k, n) and
D (m, n), or batched (l, m, k), (l, k, n) and (l, m, n). The kernel's runtime parameters are the three tensors, each a
struct of its address, its extents (int32) and its non-unit strides (int64), which Rust packs (`cuda_gemm`) and passes
to a plain ``cuLaunchKernel``, with no Python per matmul."""

import builtins
import linecache
import types

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from lumen.codegen.dot_general.cuda.ampere import AmpereGemm
from lumen.codegen.utils.cuda import VECTOR_BYTES, check, compiled_cubin, modules

# The CuTe type of each dtype a GEMM reads or writes, by lumen's name for it.
DTYPES = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}


def tensor_bytes(rank):
    """Bytes of a rank-`rank` operand's kernel parameter: its address, `rank` int32 extents padded to 8 bytes, and
    `rank` - 1 int64 strides (all but the contiguous one's)."""
    return 8 + (4 * rank + 7) // 8 * 8 + 8 * (rank - 1)


def _module(name, source):
    """The generated module `source` named `name`, executed, its source registered with linecache (CuTe reads its
    functions' source)."""
    filename = f"<lumen {name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    module = types.ModuleType(name)
    exec(builtins.compile(source, filename, "exec"), vars(module))
    return module


def _launcher(gemm, pointers, inputs, fused):
    """A jit function launching `gemm` on A, B and D (none when `fused`) and `pointers` pointers, its first `inputs`
    a fused epilogue's inputs and the rest its outputs, then the stream: each tensor and pointer one argument."""
    names = [f"p{i}" for i in range(pointers)]
    params = "".join(f"{p}, " for p in names)
    d = "None" if fused else "d"
    source = (
        f"def launch(a, b, {'' if fused else 'd, '}{params}stream):\n"
        f"    gemm(a, b, {d}, [{', '.join(names[:inputs])}], [{', '.join(names[inputs:])}], stream)\n"
    )
    filename = f"<lumen gemm launch {pointers} {inputs} {fused}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope = {"gemm": gemm}
    exec(builtins.compile(source, filename, "exec"), scope)
    return cute.jit(scope["launch"])


def compile(
    a_dtype,
    b_dtype,
    d_dtype,
    a_major,
    b_major,
    d_major,
    batched,
    tile,
    aligned_m,
    aligned_n,
    aligned_k,
    layouts,
    epilogue=None,
):
    """The GEMM for operands of dtypes named `a_dtype` and `b_dtype` writing `d_dtype`, with majors `a_major` ("m" or
    "k"), `b_major` ("k" or "n", of B as the caller's (k, n)) and `d_major` ("m" or "n"), batched or not, with `tile`
    (tile_m, tile_n, tile_k, warps, stages) from `codegen/dot_general/cuda/config.rs` (stages 0 for as many as fit),
    accumulating in float32, loaded into the current context; `aligned_m`, `aligned_n` and `aligned_k` for an m, n or
    k that is a multiple of the tile's (fewer bounds checks, see `AmpereGemm`); `layouts`, A's, B's and D's (shape,
    strides) as the caller holds them, compiles it for exactly those (each tensor then passed as its address alone).
    With `epilogue`, (name, source, aligned) of a fused epilogue's generated module (`codegen/dot_general/cuda`), it
    writes no D (`layouts` holds A's and B's alone; `d_dtype` is the product's): the epilogue computes the fusion's
    outputs from the product, its tensors passed as pointers after A and B, 16-byte aligned when `aligned`.
    Returns its launch info as `lumen.codegen.utils.cuda.compile` does: its ``CUfunction``, warps per block, dynamic
    shared bytes, that a plain ``cuLaunchKernel`` runs it, and its parameters' types (a tensor struct each, of
    `tensor_bytes` bytes, checked against the loaded kernel)."""
    if a_dtype != b_dtype or a_dtype not in ("float16", "bfloat16"):
        raise RuntimeError(
            f"dot_general: the CUDA GEMM takes two float16 or bfloat16 operands, got {a_dtype} and {b_dtype}"
        )

    if d_dtype not in DTYPES:
        raise RuntimeError(f"dot_general: the CUDA GEMM writes float16, bfloat16 or float32, got {d_dtype}")

    ab, out = DTYPES[a_dtype], DTYPES[d_dtype]
    generated = None if epilogue is None else _module(*epilogue[:2])
    aligned = epilogue is not None and epilogue[2]
    gemm = AmpereGemm(
        cutlass.Float32,
        ab,
        tile_shape_mnk=tile[:3],
        NUM_WARPS=tile[3],
        ab_stage=tile[4] or None,
        aligned_m=aligned_m,
        aligned_n=aligned_n,
        aligned_k=aligned_k,
        epilogue=generated,
        epilogue_aligned=aligned,
        d_dtype=out,
    )

    # Static shapes and strides: CuTe folds them into the kernel.
    tensors = [
        cute.runtime.make_fake_tensor(dtype, tuple(shape), stride=tuple(strides), assumed_align=16)
        for dtype, (shape, strides) in zip((ab, ab, out), layouts)
    ]

    # A fused epilogue's tensors, by pointer.
    align = VECTOR_BYTES if aligned else None
    pointers = [] if generated is None else generated.TYPES
    pointers = [make_ptr(t, VECTOR_BYTES, cute.AddressSpace.gmem, assumed_align=align) for t in pointers]
    launch = _launcher(gemm, len(pointers), 0 if generated is None else generated.INPUTS, generated is not None)

    compiled, cubin = compiled_cubin(launch, *tensors, *pointers, cute.runtime.make_fake_stream())
    (entry,) = compiled.kernel_info

    status, module = cuda.cuModuleLoadData(cubin)
    check(status, entry)
    modules.append(module)
    status, function = cuda.cuModuleGetFunction(module, entry.encode())
    check(status, entry)

    # Rust packs each tensor as CuTe lays its struct out: a mismatch must fail here, not launch on garbage.
    sizes = []
    while (info := cuda.cuFuncGetParamInfo(function, len(sizes)))[0] == cuda.CUresult.CUDA_SUCCESS:
        sizes.append(info[2])
    if sizes != [8] * (len(tensors) + len(pointers)):
        raise RuntimeError(f"dot_general: the CUDA GEMM's parameters are {sizes} bytes, not addresses")

    attribute = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    (status,) = cuda.cuFuncSetAttribute(function, attribute, gemm.shared_bytes)
    check(status, entry)

    return int(function), gemm.NUM_WARPS, gemm.shared_bytes, True, ["tensor"] * len(tensors) + ["*"] * len(pointers)
