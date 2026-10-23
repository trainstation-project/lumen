import math

import triton
import triton.language as tl


@triton.jit
def no_epilogue(D, BLOCK_M, BLOCK_N):
    return D


@triton.jit
def gemm_kernel(
    A_ptr,
    A_stride,
    B_ptr,
    B_stride,
    D_ptr,
    D_stride,
    M,
    K,
    N,
    ACCUM_DTYPE: tl.constexpr,
    EPILOGUE: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # A -> L x M x K, B -> L x K x N, D -> L x M x N, each as strided as its strides

    BLOCK_ID_L = tl.program_id(axis=0).to(tl.int64)
    BLOCK_ID = tl.program_id(axis=1)

    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_SIZE_N)

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

    MASK_M = BLOCK_M < M
    MASK_N = BLOCK_N < N

    D = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUM_DTYPE)
    BLOCK_K = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    for _ in range(tl.cdiv(K, BLOCK_SIZE_K)):
        MASK_K = BLOCK_K < K

        A_ptrs = A_ptr + BLOCK_ID_L * A_stride[0] + BLOCK_M[:, None] * A_stride[1] + BLOCK_K[None, :] * A_stride[2]
        A = tl.load(A_ptrs, mask=MASK_M[:, None] & MASK_K[None, :], other=0)

        B_ptrs = B_ptr + BLOCK_ID_L * B_stride[0] + BLOCK_K[:, None] * B_stride[1] + BLOCK_N[None, :] * B_stride[2]
        B = tl.load(B_ptrs, mask=MASK_K[:, None] & MASK_N[None, :], other=0)

        D = tl.dot(A, B, D, input_precision="ieee", out_dtype=ACCUM_DTYPE)
        BLOCK_K += BLOCK_SIZE_K

    D = EPILOGUE(D, BLOCK_M, BLOCK_N)

    tl.store(
        D_ptr + BLOCK_ID_L * D_stride[0] + BLOCK_M[:, None] * D_stride[1] + BLOCK_N[None, :] * D_stride[2],
        D,
        mask=MASK_M[:, None] & MASK_N[None, :],
    )


TL_DTYPES = {"float32": tl.float32, "float16": tl.float16, "bfloat16": tl.bfloat16}


def _merged(t, dims, what):
    dims = [d for d in dims if t.shape[d] != 1]
    for outer, inner in zip(dims, dims[1:]):
        if t.strides[outer] != t.strides[inner] * t.shape[inner]:
            raise ValueError(f"dot_general: its {what} dimensions {dims} of {list(t.shape)} are not one strided run")
    return math.prod(t.shape[d] for d in dims), t.strides[dims[-1]] if dims else 0


def dot_general(tensors, contracting, batch, accum_dtype, config, epilogue=no_epilogue):
    lhs, rhs, out = tensors
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = contracting, batch
    if accum_dtype not in TL_DTYPES or (lhs.dtype, accum_dtype) == ("bfloat16", "bfloat16"):
        raise ValueError(f"dot_general: Triton's dot accumulates in float32 or float16, got {accum_dtype}")
    lhs_free = [d for d in range(len(lhs.shape)) if d not in lhs_contracting and d not in lhs_batch]
    rhs_free = [d for d in range(len(rhs.shape)) if d not in rhs_contracting and d not in rhs_batch]
    out_dims = list(range(len(out.shape)))
    L, lhs_l = _merged(lhs, lhs_batch, "batch")
    _, rhs_l = _merged(rhs, rhs_batch, "batch")
    M, lhs_m = _merged(lhs, lhs_free, "free")
    N, rhs_n = _merged(rhs, rhs_free, "free")
    K, lhs_k = _merged(lhs, lhs_contracting, "contracting")
    _, rhs_k = _merged(rhs, rhs_contracting, "contracting")
    _, out_l = _merged(out, out_dims[: len(lhs_batch)], "batch")
    _, out_m = _merged(out, out_dims[len(lhs_batch) : len(lhs_batch) + len(lhs_free)], "free")
    _, out_n = _merged(out, out_dims[len(lhs_batch) + len(lhs_free) :], "free")
    if L == 0 or M == 0 or N == 0:
        return
    block_m, block_n, block_k, group_m, stages, warps = config
    gemm_kernel[(L, triton.cdiv(M, block_m) * triton.cdiv(N, block_n))](
        A_ptr=lhs,
        A_stride=(lhs_l, lhs_m, lhs_k),
        B_ptr=rhs,
        B_stride=(rhs_l, rhs_k, rhs_n),
        D_ptr=out,
        D_stride=(out_l, out_m, out_n),
        M=M,
        K=K,
        N=N,
        ACCUM_DTYPE=TL_DTYPES[accum_dtype],
        EPILOGUE=epilogue,
        GROUP_SIZE_M=group_m,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_K=block_k,
        BLOCK_SIZE_N=block_n,
        num_stages=stages,
        num_warps=warps,
    )
