"""Triton matrix-vector product (GEMV): a few rows times a matrix read along K, with the GEMM's generated operand
prologues and fused epilogue."""

import triton
import triton.language as tl

from lumen.codegen.devices.triton import compiled_launch, functions, scalar
from lumen.codegen.dot_general.triton.gemm_kernel import TL_DTYPES


@triton.jit
def gemv_kernel(
    A_ptr,
    B_ptr,
    D_ptr,
    EXTRA,
    M,
    K,
    N,
    ACCUM_DTYPE: tl.constexpr,
    PROLOGUE_A: tl.constexpr,
    PROLOGUE_B: tl.constexpr,
    EPILOGUE: tl.constexpr,
    GROUP: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """All ``M`` (at most ``BLOCK_SIZE_M``) rows times ``BLOCK_SIZE_N`` columns of batch ``program_id(0)``: each
    step loads ``BLOCK_SIZE_K`` of K of the rows and of those columns of the matrix (contiguous along K),
    accumulates their products, and the sum over K goes to ``EPILOGUE`` as ``gemm_kernel``'s accumulator does.
    With column halves (``GROUP > 1``, not interleaved) it also sums the second half's columns."""
    BLOCK_ID_L = tl.program_id(axis=0).to(tl.int64)
    BLOCK_ID_N = tl.program_id(axis=1)

    # With halves, the grid covers only the first half of N; the second is offset by GRID_N.
    HALVES: tl.constexpr = GROUP > 1 and not INTERLEAVED
    GRID_N = N // GROUP if HALVES else N

    BLOCK_M = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    BLOCK_N = (BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    MASK_M = BLOCK_M < M
    MASK_N = BLOCK_N < GRID_N

    # Products accumulated per lane of K, summed over K once at the end.
    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=ACCUM_DTYPE)
    if HALVES:
        D_HALF = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=ACCUM_DTYPE)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K
        A = PROLOGUE_A(A_ptr, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_K[None, :], MASK_M[:, None] & MASK_K[None, :])
        # The matrix's columns as rows, K along the lanes (its contiguous dim).
        B = PROLOGUE_B(B_ptr, BLOCK_ID_L, BLOCK_K[None, :], BLOCK_N[:, None], MASK_N[:, None] & MASK_K[None, :])
        A = A.to(ACCUM_DTYPE)[:, None, :]
        D += A * B.to(ACCUM_DTYPE)[None, :, :]

        if HALVES:
            B_HALF = PROLOGUE_B(
                B_ptr, BLOCK_ID_L, BLOCK_K[None, :], GRID_N + BLOCK_N[:, None], MASK_N[:, None] & MASK_K[None, :]
            )
            D_HALF += A * B_HALF.to(ACCUM_DTYPE)[None, :, :]

        BLOCK_K += BLOCK_SIZE_K

    MASK = MASK_M[:, None] & MASK_N[None, :]

    if HALVES:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], (tl.sum(D, 2), tl.sum(D_HALF, 2)), MASK)
    else:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], tl.sum(D, 2), MASK)


def dot_general(tensors, sizes, accum_dtype, config, specs, extra, scalars, flags, compile=False):
    """Launches ``gemv_kernel`` as ``gemm_kernel.dot_general`` launches the GEMM (the same arguments): ``config``'s
    ``block_m`` covers every row, a program per ``block_n`` columns (of the first half, with halves)."""
    lhs, rhs, out = tensors
    if accum_dtype not in TL_DTYPES or (lhs.dtype, accum_dtype) == ("bfloat16", "bfloat16"):
        raise ValueError(f"dot_general: Triton's dot accumulates in float32 or float16, got {accum_dtype}")
    L, M, K, N = sizes
    if L == 0 or M == 0 or N == 0:
        return None
    block_m, block_n, block_k, _, _, warps = config
    group, interleaved, rows = flags
    grid_n = N // group if group > 1 and not interleaved else N
    grid = (L, triton.cdiv(grid_n, block_n))
    kwargs = dict(
        A_ptr=lhs,
        B_ptr=rhs,
        D_ptr=out,
        EXTRA=tuple(scalar(x) if k in scalars else x for k, x in enumerate(extra)),
        M=M,
        K=K,
        N=N,
        ACCUM_DTYPE=TL_DTYPES[accum_dtype],
        **functions(specs),
        GROUP=group,
        INTERLEAVED=interleaved,
        ROWS=rows,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        BLOCK_SIZE_N=block_n,
        num_warps=warps,
        # The epilogue may contract a multiply and an add into one FMA, as the GEMM's.
        enable_fp_fusion=True,
    )
    if compile:
        return compiled_launch(gemv_kernel, grid, (), kwargs)
    gemv_kernel[grid](**kwargs)
