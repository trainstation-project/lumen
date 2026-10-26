import triton
import triton.language as tl

from lumen.codegen.triton import functions, scalar


@triton.jit
def gemm_kernel(
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
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    BLOCK_ID_L = tl.program_id(axis=0).to(tl.int64)
    BLOCK_ID = tl.program_id(axis=1)

    HALVES: tl.constexpr = GROUP > 1 and not INTERLEAVED
    GRID_M = M // GROUP if HALVES and ROWS else M
    GRID_N = N // GROUP if HALVES and not ROWS else N
    NUM_BLOCKS_M = tl.cdiv(GRID_M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(GRID_N, BLOCK_SIZE_N)

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

    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)
    if HALVES:
        D_HALF = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K
        A = PROLOGUE_A(A_ptr, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_K[None, :], MASK_M[:, None] & MASK_K[None, :])
        B = PROLOGUE_B(B_ptr, BLOCK_ID_L, BLOCK_K[:, None], BLOCK_N[None, :], MASK_K[:, None] & MASK_N[None, :])
        D = tl.dot(A, B, D, input_precision="ieee", out_dtype=ACCUM_DTYPE)

        if HALVES:
            if ROWS:
                A_HALF = PROLOGUE_A(
                    A_ptr, BLOCK_ID_L, GRID_M + BLOCK_M[:, None], BLOCK_K[None, :], MASK_M[:, None] & MASK_K[None, :]
                )
                D_HALF = tl.dot(A_HALF, B, D_HALF, input_precision="ieee", out_dtype=ACCUM_DTYPE)
            else:
                B_HALF = PROLOGUE_B(
                    B_ptr, BLOCK_ID_L, BLOCK_K[:, None], GRID_N + BLOCK_N[None, :], MASK_K[:, None] & MASK_N[None, :]
                )
                D_HALF = tl.dot(A, B_HALF, D_HALF, input_precision="ieee", out_dtype=ACCUM_DTYPE)

        BLOCK_K += BLOCK_SIZE_K

    MASK = MASK_M[:, None] & MASK_N[None, :]

    if HALVES:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], (D, D_HALF), MASK)
    else:
        EPILOGUE(D_ptr, EXTRA, BLOCK_ID_L, BLOCK_M[:, None], BLOCK_N[None, :], D, MASK)


TL_DTYPES = {"float32": tl.float32, "float16": tl.float16, "bfloat16": tl.bfloat16}


def dot_general(tensors, sizes, accum_dtype, config, specs, extra, scalars, flags):
    lhs, rhs, out = tensors
    if accum_dtype not in TL_DTYPES or (lhs.dtype, accum_dtype) == ("bfloat16", "bfloat16"):
        raise ValueError(f"dot_general: Triton's dot accumulates in float32 or float16, got {accum_dtype}")
    L, M, K, N = sizes
    if L == 0 or M == 0 or N == 0:
        return
    block_m, block_n, block_k, group_m, stages, warps = config
    group, interleaved, rows = flags
    halves = group > 1 and not interleaved
    grid_m, grid_n = (M // group if halves and rows else M), (N // group if halves and not rows else N)
    gemm_kernel[(L, triton.cdiv(grid_m, block_m) * triton.cdiv(grid_n, block_n))](
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
        GROUP_SIZE_M=group_m,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        BLOCK_SIZE_N=block_n,
        num_stages=stages,
        num_warps=warps,
    )
