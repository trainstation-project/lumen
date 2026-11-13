"""Compiles lumen's flash-attention forward fusions on CUDA through the Ampere kernel
(`ampere.forward.AmpereAttentionForward`) for Rust to launch, as the CUDA GEMM compiles
(`lumen.codegen.dot_general.cuda.launch`): once per operand layout, score ops, causal diagonal and config, against
fake tensors of those exact shapes and strides, so each tensor parameter is its address alone and every launch is a
plain ``cuLaunchKernel`` from Rust (`cuda_attention`)."""

import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from lumen.codegen.attention.cuda.ampere.forward import AmpereAttentionForward
from lumen.codegen.utils.cuda import check, modules

# The CuTe type of each dtype the kernel reads or writes, by lumen's name for it.
DTYPES = {
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
}


def rounded(c, dtype):
    """Float `c` rounded to nearest even in lumen dtype `dtype` (float16, bfloat16 or float32), as a Python float."""
    if dtype == "float16":
        return struct.unpack("<e", struct.pack("<e", c))[0]

    bits = struct.unpack("<I", struct.pack("<f", c))[0]

    if dtype == "bfloat16":
        # Round the float32 bits' low half to nearest even (NaN and inf pass through, their low half dropped).
        if (bits >> 23) & 0xFF != 0xFF:
            bits += 0x7FFF + ((bits >> 16) & 1)
        bits &= 0xFFFF0000

    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def compile(dtype, layouts, heads_per_kv, causal, scores, config):
    """The forward for float16 or bfloat16 `dtype` Q, K, V and O, and a float32 LSE when `layouts` has five entries,
    loaded into the current context. `layouts` gives each operand's (shape, strides) in the kernel's order: Q, K, V
    and O as (batch, rows, heads, head dim), the LSE as (batch, heads, rows), every head dim contiguous; `heads_per_kv`
    query heads share each K/V head. `causal` is the causal diagonal's shift (key `k` seen by query `q` when
    `k <= q + causal`) or None; `scores` the traced score ops, `("round", _, dtype)` or `("mul", c, dtype)` with
    lumen dtype names; `config` (tile_m, tile_n, stages, warps, q_in_regs) from `codegen/attention/cuda/config.rs`.
    Returns its launch info as `lumen.codegen.utils.cuda.compile` does: its ``CUfunction``, warps per block, dynamic
    shared bytes, that a plain ``cuLaunchKernel`` runs it, and its parameters' types (each an address)."""
    if dtype not in ("float16", "bfloat16"):
        raise RuntimeError(f"flash_attention: the CUDA kernel reads float16 or bfloat16, got {dtype}")

    tile_m, tile_n, stages, warps, q_in_regs = config
    (_, _, _, h), (_, _, _, hv) = layouts[0][0], layouts[2][0]
    ops = tuple(
        ("round", DTYPES[d]) if kind == "round" else ("mul", rounded(c, d), DTYPES[d]) for kind, c, d in scores
    )

    attention = AmpereAttentionForward(
        DTYPES[dtype],
        h,
        hv,
        qhead_per_kvhead=heads_per_kv,
        is_causal=causal is not None,
        softmax_scale=1.0,
        causal_shift=causal,
        scores=ops,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=stages,
        num_threads=32 * warps,
        Q_in_regs=q_in_regs,
    )

    types = [DTYPES[dtype]] * 4 + [cutlass.Float32]
    tensors = [
        cute.runtime.make_fake_tensor(ty, tuple(shape), stride=tuple(strides), assumed_align=16 if k < 4 else 4)
        for k, (ty, (shape, strides)) in enumerate(zip(types, layouts))
    ]
    lse = tensors[4] if len(tensors) == 5 else None

    compiled = cute.compile(
        attention,
        *tensors[:4],
        lse,
        cute.runtime.make_fake_stream(),
        options="--keep-cubin",
    )
    (entry,) = compiled.kernel_info

    status, module = cuda.cuModuleLoadData(compiled.artifacts.CUBIN)
    check(status, entry)
    modules.append(module)
    status, function = cuda.cuModuleGetFunction(module, entry.encode())
    check(status, entry)

    # Rust passes each tensor as its address: a mismatch must fail here, not launch on garbage.
    sizes = []
    while (info := cuda.cuFuncGetParamInfo(function, len(sizes)))[0] == cuda.CUresult.CUDA_SUCCESS:
        sizes.append(info[2])
    if sizes != [8] * len(layouts):
        raise RuntimeError(
            f"flash_attention: the CUDA kernel's parameters are {sizes} bytes, not {len(layouts)} addresses"
        )

    attribute = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
    (status,) = cuda.cuFuncSetAttribute(function, attribute, attention.shared_bytes)
    check(status, entry)

    return int(function), warps, attention.shared_bytes, True, ["tensor"] * len(layouts)
