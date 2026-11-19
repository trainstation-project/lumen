"""Triton split-K GEMM: a kernel writing each K chunk's product of an output block, then one summing the chunks
in order (deterministic) and running the GEMM's fused epilogue on the sum."""

import triton
import triton.language as tl

from lumen.codegen.dot_general.triton.gemm_kernel import TL_DTYPES
from lumen.codegen.utils.triton import compiled_launch, functions, scalar


@triton.jit
def partial(P_ptr, chunk, l, L, M, N, m, n):
    """Where chunk ``chunk``'s product of batch ``l`` (of ``L``) at row ``m`` and column ``n`` lies in the
    ``[SPLIT, L, M, N]`` partials ``P_ptr``."""
    return P_ptr + ((chunk * L + l) * M + m) * N + n


@triton.jit
def partials_kernel(
    A_ptr,
    B_ptr,
    P_ptr,
    M,
    K,
    N,
    ACCUM_DTYPE: tl.constexpr,
    PROLOGUE_A: tl.constexpr,
    PROLOGUE_B: tl.constexpr,
    GROUP: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROWS: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One ``BLOCK_SIZE_M x BLOCK_SIZE_N`` output block's product over one of ``SPLIT`` K chunks
    (``program_id(0)`` is the batch times ``SPLIT`` plus the chunk), stored to the partials. Its operands load
    through ``PROLOGUE_A``/``PROLOGUE_B`` and its blocks are ordered as ``gemm_kernel``'s; with row or column
    halves (``GROUP > 1``, not interleaved) it also stores the second half's product."""
    BLOCK_ID_L = (tl.program_id(axis=0) // SPLIT).to(tl.int64)
    CHUNK = tl.program_id(axis=0) % SPLIT
    L = tl.num_programs(axis=0) // SPLIT
    BLOCK_ID = tl.program_id(axis=1)

    # With halves, the grid covers only the first half of M (ROWS) or N; the second is offset by GRID_M/GRID_N.
    HALVES: tl.constexpr = GROUP > 1 and not INTERLEAVED
    GRID_M = M // GROUP if HALVES and ROWS else M
    GRID_N = N // GROUP if HALVES and not ROWS else N
    NUM_BLOCKS_M = tl.cdiv(GRID_M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(GRID_N, BLOCK_SIZE_N)

    # Grouped ordering: GROUP_SIZE_M row blocks share each column sweep, for L2 reuse.
    NUM_BLOCKS_IN_GROUP = GROUP_SIZE_M * NUM_BLOCKS_N
    GROUP_ID = BLOCK_ID // NUM_BLOCKS_IN_GROUP

    FIRST_BLOCK_M_IN_GROUP = GROUP_ID * GROUP_SIZE_M
    CURRENT_GROUP_SIZE_M = min(NUM_BLOCKS_M - FIRST_BLOCK_M_IN_GROUP, GROUP_SIZE_M)

    BLOCK_ID_M = FIRST_BLOCK_M_IN_GROUP + ((BLOCK_ID % NUM_BLOCKS_IN_GROUP) % CURRENT_GROUP_SIZE_M)
    BLOCK_ID_N = (BLOCK_ID % NUM_BLOCKS_IN_GROUP) // CURRENT_GROUP_SIZE_M

    if BLOCK_ID_N >= NUM_BLOCKS_N:
        return

    BLOCK_M = (BLOCK_ID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    BLOCK_N = (BLOCK_ID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)

    MASK_M = BLOCK_M < GRID_M
    MASK_N = BLOCK_N < GRID_N

    # The second half's block: below the first (ROWS) or right of it.
    if ROWS:
        M_HALF, N_HALF = GRID_M + BLOCK_M, BLOCK_N
    else:
        M_HALF, N_HALF = BLOCK_M, GRID_N + BLOCK_N

    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)
    if HALVES:
        D_HALF = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)

    # This program's K chunk.
    K_END = (CHUNK + 1) * (K // SPLIT)
    BLOCK_K = CHUNK * (K // SPLIT) + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for _ in range(tl.cdiv(K // SPLIT, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K_END
        A = PROLOGUE_A(A_ptr, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_K[None, :], MASK_M[:, None] & MASK_K[None, :])
        B = PROLOGUE_B(B_ptr, BLOCK_ID_L, BLOCK_K[:, None], BLOCK_N[None, :], MASK_K[:, None] & MASK_N[None, :])
        D = tl.dot(A, B, D, input_precision="ieee", out_dtype=ACCUM_DTYPE)

        if HALVES:
            if ROWS:
                A_HALF = PROLOGUE_A(
                    A_ptr, BLOCK_ID_L, M_HALF[:, None], BLOCK_K[None, :], MASK_M[:, None] & MASK_K[None, :]
                )
                D_HALF = tl.dot(A_HALF, B, D_HALF, input_precision="ieee", out_dtype=ACCUM_DTYPE)
            else:
                B_HALF = PROLOGUE_B(
                    B_ptr, BLOCK_ID_L, BLOCK_K[:, None], N_HALF[None, :], MASK_K[:, None] & MASK_N[None, :]
                )
                D_HALF = tl.dot(A, B_HALF, D_HALF, input_precision="ieee", out_dtype=ACCUM_DTYPE)

        BLOCK_K += BLOCK_SIZE_K

    MASK = MASK_M[:, None] & MASK_N[None, :]
    tl.store(partial(P_ptr, CHUNK, BLOCK_ID_L, L, M, N, BLOCK_M[:, None], BLOCK_N[None, :]), D, mask=MASK)

    if HALVES:
        tl.store(partial(P_ptr, CHUNK, BLOCK_ID_L, L, M, N, M_HALF[:, None], N_HALF[None, :]), D_HALF, mask=MASK)


@triton.jit
def combine_kernel(
    P_ptr,
    D_ptr,
    EXTRA,
    M,
    N,
    ACCUM_DTYPE: tl.constexpr,
    EPILOGUE: tl.constexpr,
    GROUP: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One ``BLOCK_SIZE_M x BLOCK_SIZE_N`` output block of batch ``program_id(0)``: the ``SPLIT`` chunks'
    partials summed in chunk order and handed to ``EPILOGUE`` as ``gemm_kernel`` hands its accumulator (both
    halves' with row or column halves)."""
    BLOCK_ID_L = tl.program_id(axis=0).to(tl.int64)
    L = tl.num_programs(axis=0)
    BLOCK_ID = tl.program_id(axis=1)

    HALVES: tl.constexpr = GROUP > 1 and not INTERLEAVED
    GRID_M = M // GROUP if HALVES and ROWS else M
    GRID_N = N // GROUP if HALVES and not ROWS else N
    NUM_BLOCKS_N = tl.cdiv(GRID_N, BLOCK_SIZE_N)

    BLOCK_M = ((BLOCK_ID // NUM_BLOCKS_N) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    BLOCK_N = ((BLOCK_ID % NUM_BLOCKS_N) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    MASK = (BLOCK_M < GRID_M)[:, None] & (BLOCK_N < GRID_N)[None, :]

    if ROWS:
        M_HALF, N_HALF = GRID_M + BLOCK_M, BLOCK_N
    else:
        M_HALF, N_HALF = BLOCK_M, GRID_N + BLOCK_N

    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)
    if HALVES:
        D_HALF = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)

    for c in range(SPLIT):
        D += tl.load(partial(P_ptr, c, BLOCK_ID_L, L, M, N, BLOCK_M[:, None], BLOCK_N[None, :]), mask=MASK)

        if HALVES:
            D_HALF += tl.load(partial(P_ptr, c, BLOCK_ID_L, L, M, N, M_HALF[:, None], N_HALF[None, :]), mask=MASK)

    if HALVES:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], (D, D_HALF), MASK)
    else:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], D, MASK)


def split_k(tensors, sizes, accum_dtype, config, specs, extra, scalars, flags, split, compile=False):
    """Launches one of a split-K GEMM's kernels on ``tensors`` (lhs, rhs, out) with ``sizes`` (L, M, K, N),
    ``split`` being ``(chunks, partials, combine)``: ``partials_kernel`` writing each of ``chunks`` K chunks'
    products to ``partials`` with ``config``'s tile, or, ``combine`` a ``(rows, columns)`` block,
    ``combine_kernel`` summing them through the epilogue a block a program. ``specs``, ``extra``, ``scalars`` and ``flags`` are ``gemm_kernel.dot_general``'s. With
    ``compile``, returns its ``compiled_launch`` instead of launching it."""
    lhs, rhs, out = tensors
    if accum_dtype not in TL_DTYPES or (lhs.dtype, accum_dtype) == ("bfloat16", "bfloat16"):
        raise ValueError(f"dot_general: Triton's dot accumulates in float32 or float16, got {accum_dtype}")
    L, M, K, N = sizes
    if L == 0 or M == 0 or N == 0:
        return None
    chunks, partials, combine = split
    group, interleaved, rows = flags
    fns = functions(specs)
    halves = group > 1 and not interleaved
    grid_m, grid_n = (M // group if halves and rows else M), (N // group if halves and not rows else N)
    if combine:
        block_m, block_n = combine
        kernel, grid = combine_kernel, (L, triton.cdiv(grid_m, block_m) * triton.cdiv(grid_n, block_n))
        kwargs = dict(
            P_ptr=partials,
            D_ptr=out,
            EXTRA=tuple(scalar(x) if k in scalars else x for k, x in enumerate(extra)),
            M=M,
            N=N,
            EPILOGUE=fns["EPILOGUE"],
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            num_warps=4,
        )
    else:
        block_m, block_n, block_k, group_m, stages, warps = config
        kernel, grid = partials_kernel, (L * chunks, triton.cdiv(grid_m, block_m) * triton.cdiv(grid_n, block_n))
        kwargs = dict(
            A_ptr=lhs,
            B_ptr=rhs,
            P_ptr=partials,
            M=M,
            K=K,
            N=N,
            PROLOGUE_A=fns["PROLOGUE_A"],
            PROLOGUE_B=fns["PROLOGUE_B"],
            GROUP_SIZE_M=group_m,
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_N=block_n,
            num_stages=stages,
            num_warps=warps,
        )
    kwargs.update(
        ACCUM_DTYPE=TL_DTYPES[accum_dtype],
        GROUP=group,
        INTERLEAVED=interleaved,
        ROWS=rows,
        SPLIT=chunks,
        # The epilogue may contract a multiply and an add into one FMA, as `gemm_kernel`'s.
        enable_fp_fusion=True,
    )
    if compile:
        return compiled_launch(kernel, grid, (), kwargs)
    kernel[grid](**kwargs)
