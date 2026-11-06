"""Runs lumen's matmuls on CUDA through the Ampere GEMM (`gemm_sm80.AmpereGemm`), the way QuACK hosts its GEMMs: compiled
once per dtypes, majors and batching against fake tensors whose extents and strides are symbolic (only each operand's
contiguous dimension static, its other strides 16-byte multiples), then called on lumen tensors through DLPack, each
passed as it is: A (m, k), B (k, n) and D (m, n), or batched (l, m, k), (l, k, n) and (l, m, n)."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from lumen.codegen.dot_general.cuda.gemm_sm80 import AmpereGemm
from lumen.kernels.cuda.compile_utils import div_for_dtype, fake_batched

# The CuTe type of each dtype a GEMM reads or writes, by lumen's name for it.
DTYPES = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}


def _major(t, rows, columns):
    """The label of `t`'s contiguous dimension: `columns` when it is its last, else `rows` (QuACK's `_get_major`)."""
    return columns if t.strides[-1] == 1 else rows


# CTA tiles (M, N, K): 128 x 256 (8 warps of 64 x 64) when the problem fills the GPU with them, else 128 x 128 (4 warps,
# two CTAs per SM) so smaller problems still spread over every SM.
WIDE_TILE, NARROW_TILE = (128, 256, 32), (128, 128, 32)


@functools.lru_cache(maxsize=None)
def _num_sms(device):
    """The SM count of CUDA device `device` (lumen's name for it, "cuda:<index>")."""
    cuda.cuInit(0)
    _, dev = cuda.cuDeviceGet(int(device.rpartition(":")[2] or 0))
    return cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)[1]


def _tile(d):
    """The CTA tile for output `d` (m, n) or (l, m, n): the wide tile once it alone makes a wave of CTAs."""
    *batch, m, n = d.shape
    tiles = (batch[0] if batch else 1) * -(-m // WIDE_TILE[0]) * -(-n // WIDE_TILE[1])
    return WIDE_TILE if tiles >= _num_sms(str(d.device)) else NARROW_TILE


@functools.lru_cache(maxsize=None)
def compiled(a_dtype, b_dtype, d_dtype, a_major, b_major, d_major, batched, tile=NARROW_TILE):
    """The GEMM compiled for operands of dtypes named `a_dtype`, `b_dtype` and `d_dtype` with majors `a_major` ("m" or
    "k"), `b_major` ("k" or "n", of B as the caller's (k, n)) and `d_major` ("m" or "n"), batched or not, with CTA tile
    `tile`, accumulating in float32."""
    if a_dtype != b_dtype or a_dtype not in ("float16", "bfloat16"):
        raise RuntimeError(
            f"dot_general: the CUDA GEMM takes two float16 or bfloat16 operands, got {a_dtype} and {b_dtype}"
        )

    if d_dtype not in DTYPES:
        raise RuntimeError(f"dot_general: the CUDA GEMM writes float16, bfloat16 or float32, got {d_dtype}")

    ab, out = DTYPES[a_dtype], DTYPES[d_dtype]
    gemm = AmpereGemm(cutlass.Float32, ab, tile_shape_mnk=tile)

    # One symbolic extent per problem dimension, shared by the operands that have it.
    m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
    l = cute.sym_int() if batched else None  # noqa: E741

    a = fake_batched(ab, m, k, l, 1 if a_major == "k" else 0, div_for_dtype(ab))
    b = fake_batched(ab, k, n, l, 1 if b_major == "n" else 0, div_for_dtype(ab))
    d = fake_batched(out, m, n, l, 1 if d_major == "n" else 0, div_for_dtype(out))

    return cute.compile(gemm, a, b, d, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi")


def matmul(a, b, d, stream):
    """Writes ``a @ b`` into ``d``, lumen tensors on one CUDA device, (m, k), (k, n) and (m, n) or batched (l, m, k),
    (l, k, n) and (l, m, n), on CUDA stream `stream` (its handle as an int)."""
    majors = _major(a, "m", "k"), _major(b, "k", "n"), _major(d, "m", "n")
    gemm = compiled(a.dtype, b.dtype, d.dtype, *majors, len(a.shape) == 3, _tile(d))
    gemm(a, b, d, cuda.CUstream(stream))
