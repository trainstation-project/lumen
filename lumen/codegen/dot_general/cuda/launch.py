"""Runs lumen's matmuls on CUDA through the Ampere GEMM (`gemm_sm80.AmpereGemm`), the way QuACK hosts its GEMMs: compiled
once per dtypes, majors and batching against fake tensors whose extents and strides are symbolic (only each operand's
contiguous dimension static, its other strides 16-byte multiples), then called on lumen tensors through DLPack, each
passed as it is: A (m, k), B (k, n) and D (m, n), or batched (l, m, k), (l, k, n) and (l, m, n)."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from lumen.codegen.dot_general.cuda.gemm_sm80 import AmpereGemm

# The CuTe type of each dtype a GEMM reads or writes, by lumen's name for it.
DTYPES = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}


class _Ordered:
    """A lumen tensor as the GEMM reads it: exported through DLPack without waiting for the device, since the GEMM
    launches on lumen's stream, after the work writing its operands (a wait would block every launch on the queue, and
    can't be issued while a CUDA graph captures or the autotuner holds the stream)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        """Wraps `tensor`."""
        self.tensor = tensor

    def __dlpack__(self, *, stream=None, max_version=None, **kwargs):
        """The tensor's capsule, exported without synchronizing."""
        return self.tensor.__dlpack__(stream=-1, max_version=max_version)

    def __dlpack_device__(self):
        """The tensor's DLPack device."""
        return self.tensor.__dlpack_device__()


def _fake(dtype, rows, columns, batch, contiguous):
    """A fake (`batch`, `rows`, `columns`) tensor of CuTe type `dtype`, or (`rows`, `columns`) when `batch` is None, to
    compile against: dimension `contiguous` of (rows, columns) has stride 1, every other stride is a symbolic multiple of
    16 bytes, and it starts 16-byte aligned."""
    shape = (rows, columns) if batch is None else (batch, rows, columns)
    contiguous += len(shape) - 2
    vector = 128 // dtype.width
    stride = tuple(1 if i == contiguous else cute.sym_int64(divisibility=vector) for i in range(len(shape)))
    return cute.runtime.make_fake_tensor(dtype, shape, stride=stride, assumed_align=16)


def _major(t, rows, columns):
    """The label of `t`'s contiguous dimension: `columns` when it is its last, else `rows` (QuACK's `_get_major`)."""
    return columns if t.strides[-1] == 1 else rows


@functools.lru_cache(maxsize=None)
def compiled(a_dtype, b_dtype, d_dtype, a_major, b_major, d_major, batched, tile):
    """The GEMM compiled for operands of dtypes named `a_dtype`, `b_dtype` and `d_dtype` with majors `a_major` ("m" or
    "k"), `b_major` ("k" or "n", of B as the caller's (k, n)) and `d_major` ("m" or "n"), batched or not, with `tile`
    (tile_m, tile_n, tile_k, warps) from `codegen/dot_general/cuda/config.rs`, accumulating in float32."""
    if a_dtype != b_dtype or a_dtype not in ("float16", "bfloat16"):
        raise RuntimeError(
            f"dot_general: the CUDA GEMM takes two float16 or bfloat16 operands, got {a_dtype} and {b_dtype}"
        )

    if d_dtype not in DTYPES:
        raise RuntimeError(f"dot_general: the CUDA GEMM writes float16, bfloat16 or float32, got {d_dtype}")

    ab, out = DTYPES[a_dtype], DTYPES[d_dtype]
    gemm = AmpereGemm(cutlass.Float32, ab, tile_shape_mnk=tile[:3], NUM_WARPS=tile[3])

    # One symbolic extent per problem dimension, shared by the operands that have it.
    m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    l = cute.sym_int() if batched else None  # noqa: E741

    a = _fake(ab, m, k, l, 1 if a_major == "k" else 0)
    b = _fake(ab, k, n, l, 1 if b_major == "n" else 0)
    d = _fake(out, m, n, l, 1 if d_major == "n" else 0)

    return cute.compile(gemm, a, b, d, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi")


def matmul(a, b, d, stream, tile):
    """Writes ``a @ b`` into ``d``, lumen tensors on one CUDA device, (m, k), (k, n) and (m, n) or batched (l, m, k),
    (l, k, n) and (l, m, n), on CUDA stream `stream` (its handle as an int) with `tile` (tile_m, tile_n, tile_k, warps),
    the config Rust picked (tuned or untuned)."""
    majors = _major(a, "m", "k"), _major(b, "k", "n"), _major(d, "m", "n")
    gemm = compiled(a.dtype, b.dtype, d.dtype, *majors, len(a.shape) == 3, tuple(tile))
    gemm(_Ordered(a), _Ordered(b), _Ordered(d), cuda.CUstream(stream))
