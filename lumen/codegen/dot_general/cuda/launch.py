"""Runs lumen's matmuls on CUDA through CUTLASS's Ampere GEMM (`gemm_sm80`): compiled once per dtypes and layouts
(any M, N and K whose contiguous dimensions are 16-byte multiples), then called on lumen tensors through DLPack."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from lumen.codegen.dot_general.cuda.gemm_sm80 import TensorOpGemm

# The CuTe type of each dtype a GEMM reads or writes, by lumen's name for it.
DTYPES = {"float16": cutlass.Float16, "bfloat16": cutlass.BFloat16, "float32": cutlass.Float32}

# Bytes each operand's contiguous dimension and start must be a multiple of: the kernel moves 16-byte vectors.
VECTOR_BYTES = 16


@cute.jit
def _matmul(gemm: cutlass.Constexpr, a: cute.Tensor, b: cute.Tensor, d: cute.Tensor, stream: cuda.CUstream):
    """``d = a @ b`` for ``a`` (m, k), ``b`` (k, n) and ``d`` (m, n), viewed as the kernel's (m, k, l), (n, k, l)
    and (m, n, l) with one batch."""
    a = cute.make_tensor(a.iterator, cute.make_layout((*a.layout.shape, 1), stride=(*a.layout.stride, 0)))
    b = cute.make_tensor(
        b.iterator,
        cute.make_layout(
            (b.layout.shape[1], b.layout.shape[0], 1), stride=(b.layout.stride[1], b.layout.stride[0], 0)
        ),
    )
    d = cute.make_tensor(d.iterator, cute.make_layout((*d.layout.shape, 1), stride=(*d.layout.stride, 0)))
    gemm(a, b, d, stream)


def _operand(dtype, contiguous_last):
    """A fake 2-D tensor of `dtype` with symbolic extents, contiguous along its last dimension when
    `contiguous_last`, else its first, that dimension a multiple of a 16-byte vector."""
    vector = VECTOR_BYTES * 8 // dtype.width
    rows = cute.sym_int(divisibility=1 if contiguous_last else vector)
    columns = cute.sym_int(divisibility=vector if contiguous_last else 1)
    order = (1, 0) if contiguous_last else (0, 1)
    return make_fake_compact_tensor(dtype, (rows, columns), stride_order=order, assumed_align=VECTOR_BYTES)


@functools.lru_cache(maxsize=None)
def compiled(a_dtype, b_dtype, d_dtype, a_row_major, b_row_major, d_row_major):
    """The GEMM compiled for operands of dtypes named `a_dtype`, `b_dtype` and `d_dtype`, each row-major (its last
    dimension contiguous) or column-major as given, accumulating in float32."""
    if a_dtype != b_dtype or a_dtype not in ("float16", "bfloat16"):
        raise RuntimeError(
            f"dot_general: the CUDA GEMM takes two float16 or bfloat16 operands, got {a_dtype} and {b_dtype}"
        )

    if d_dtype not in DTYPES:
        raise RuntimeError(f"dot_general: the CUDA GEMM writes float16, bfloat16 or float32, got {d_dtype}")

    ab, out = DTYPES[a_dtype], DTYPES[d_dtype]
    gemm = TensorOpGemm(ab, out, cutlass.Float32, (2, 2, 1), is_m_major_c=not d_row_major)

    a = _operand(ab, a_row_major)
    b = _operand(ab, b_row_major)
    d = _operand(out, d_row_major)
    return cute.compile(_matmul, gemm, a, b, d, make_fake_stream(), options="--enable-tvm-ffi")


def _row_major(t):
    """Whether 2-D tensor `t` is contiguous along its last dimension (else along its first)."""
    return t.strides[1] == 1


def matmul(a, b, d, stream):
    """Writes ``a @ b`` into ``d`` (lumen tensors (m, k), (k, n) and (m, n) on one CUDA device) on CUDA stream
    `stream` (its handle as an int)."""
    gemm = compiled(a.dtype, b.dtype, d.dtype, _row_major(a), _row_major(b), _row_major(d))
    gemm(a, b, d, cuda.CUstream(stream))
