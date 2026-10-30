"""Triton flash attention (FlashAttention-2): a forward kernel writing each query row's log-sum-exp, and
backward kernels for dK with dV and for dQ that recompute the probabilities from it. Operands are read and
written through generated helpers that know their layouts (``LOAD_Q(ptr, b, rows, cols, mask)``,
``STORE_O(ptr, EXTRA, b, rows, cols, value, mask)``, applying the fused epilogue to the tensors in ``EXTRA``),
and ``SCORE`` applies the traced score ops (casts, scales)."""

import triton
import triton.language as tl

from lumen.codegen.triton import compiled_launch, functions


@triton.jit
def _seen(rows, keys, SQ, SK, shift, CAUSAL: tl.constexpr):
    """Which `(query, key)` pairs of the tile count: both in range and, when causal, the key on or before
    the query's diagonal `query + shift`."""
    seen = (rows < SQ)[:, None] & (keys < SK)[None, :]
    if CAUSAL:
        seen = seen & (keys[None, :] <= rows[:, None] + shift)
    return seen


@triton.jit
def _keys_end(start, SQ, SK, shift, BLOCK_SIZE_M: tl.constexpr, CAUSAL: tl.constexpr):
    """One past the last key the queries of the block at `start` see."""
    end = SK
    if CAUSAL:
        end = tl.minimum(tl.maximum(tl.minimum(start + BLOCK_SIZE_M, SQ) + shift, 0), SK)
    return end


@triton.jit
def forward_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    OFFSET,
    EXTRA,
    SQ,
    SK,
    SHIFT,
    LOAD_Q: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    STORE_O: tl.constexpr,
    STORE_LSE: tl.constexpr,
    SCORE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    HAS_LSE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """``BLOCK_SIZE_M`` queries of batch ``BLOCK_ID_B``: the keys in blocks of ``BLOCK_SIZE_N``, an online
    softmax of their scores (running max ``m`` and sum ``l``) and its probabilities, rounded to V's dtype,
    times V; normalized after the last block. A causal block stops at the last key its last query sees
    (``SHIFT`` plus ``OFFSET[0]`` past the diagonal)."""
    BLOCK_ID_M = tl.program_id(axis=0)
    BLOCK_ID_B = tl.program_id(axis=1).to(tl.int64)

    start = BLOCK_ID_M * BLOCK_SIZE_M
    BLOCK_M = (start + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H).to(tl.int64)
    BLOCK_HV = tl.arange(0, BLOCK_SIZE_HV).to(tl.int64)

    MASK_M = BLOCK_M < SQ
    MASK_H = BLOCK_H < H
    MASK_HV = BLOCK_HV < HV

    q = LOAD_Q(Q, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_H[None, :], MASK_M[:, None] & MASK_H[None, :])
    shift = SHIFT
    if HAS_OFFSET:
        shift += tl.load(OFFSET).to(tl.int64)
    m = tl.full([BLOCK_SIZE_M], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_SIZE_M], tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_HV], tl.float32)
    for k0 in range(0, _keys_end(start, SQ, SK, shift, BLOCK_SIZE_M, CAUSAL), BLOCK_SIZE_N):
        BLOCK_N = (k0 + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
        MASK_N = BLOCK_N < SK
        k = LOAD_K(K, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_H[None, :], MASK_N[:, None] & MASK_H[None, :])
        s = SCORE(tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32))
        s = tl.where(_seen(BLOCK_M, BLOCK_N, SQ, SK, shift, CAUSAL), s, float("-inf"))
        # Rescale the running sum and output to the new running max; rows that have seen no key keep 0.
        m_new = tl.maximum(m, tl.max(s, 1))
        empty = m_new == float("-inf")
        alpha = tl.where(empty, 1.0, tl.exp(m - m_new))
        p = tl.where(empty[:, None], 0.0, tl.exp(s - m_new[:, None]))
        l = l * alpha + tl.sum(p, 1)
        v = LOAD_V(V, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_HV[None, :], MASK_N[:, None] & MASK_HV[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee", out_dtype=tl.float32)
        m = m_new
    if HAS_LSE:
        STORE_LSE(LSE, BLOCK_ID_B, BLOCK_M, 0, m + tl.log(l), MASK_M)
    MASK = MASK_M[:, None] & MASK_HV[None, :]
    STORE_O(O, EXTRA, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_HV[None, :], acc / l[:, None], MASK)


@triton.jit
def dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    DQ,
    SQ,
    SK,
    SHIFT,
    SCALE,
    LOAD_Q: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    LOAD_DO: tl.constexpr,
    LOAD_LSE: tl.constexpr,
    LOAD_DELTA: tl.constexpr,
    STORE_DK: tl.constexpr,
    STORE_DV: tl.constexpr,
    SCORE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ADD_DQ: tl.constexpr = None,
):
    """dK and dV of ``BLOCK_SIZE_N`` keys of batch ``BLOCK_ID_B``, summed over the blocks of
    ``BLOCK_SIZE_M`` queries that see them, on transposed scores ``S^T = K Q^T``: ``P^T = exp(S^T - lse)``,
    ``dV += P^T dO``, ``dS^T = P^T (V dO^T - delta) SCALE`` and ``dK += dS^T Q``. With ``ADD_DQ``, also adds
    each block's ``dS K`` to the zeroed float32 ``DQ`` atomically (summed in no fixed order)."""
    BLOCK_ID_N = tl.program_id(axis=0)
    BLOCK_ID_B = tl.program_id(axis=1).to(tl.int64)

    start = BLOCK_ID_N * BLOCK_SIZE_N
    BLOCK_N = (start + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H).to(tl.int64)
    BLOCK_HV = tl.arange(0, BLOCK_SIZE_HV).to(tl.int64)

    MASK_N = BLOCK_N < SK
    MASK_H = BLOCK_H < H
    MASK_HV = BLOCK_HV < HV

    k_mask = MASK_N[:, None] & MASK_H[None, :]
    v_mask = MASK_N[:, None] & MASK_HV[None, :]
    k = LOAD_K(K, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_H[None, :], k_mask)
    v = LOAD_V(V, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_HV[None, :], v_mask)
    dk_acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_H], tl.float32)
    dv_acc = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_HV], tl.float32)
    # A causal key block starts at the first query block that sees its first key.
    first = 0
    if CAUSAL:
        first = tl.maximum(start - SHIFT, 0) // BLOCK_SIZE_M * BLOCK_SIZE_M
    for q0 in range(first, SQ, BLOCK_SIZE_M):
        BLOCK_M = (q0 + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        MASK_M = BLOCK_M < SQ
        q = LOAD_Q(Q, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_H[None, :], MASK_M[:, None] & MASK_H[None, :])
        do = LOAD_DO(DO, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_HV[None, :], MASK_M[:, None] & MASK_HV[None, :])
        lse = LOAD_LSE(LSE, BLOCK_ID_B, 0, BLOCK_M, MASK_M)
        delta = LOAD_DELTA(DELTA, BLOCK_ID_B, 0, BLOCK_M, MASK_M)
        st = SCORE(tl.dot(k, tl.trans(q), input_precision="ieee", out_dtype=tl.float32))
        seen = MASK_N[:, None] & MASK_M[None, :]
        if CAUSAL:
            seen = seen & (BLOCK_N[:, None] <= BLOCK_M[None, :] + SHIFT)
        pt = tl.where(seen, tl.exp(st - lse[None, :]), 0.0)
        dv_acc = tl.dot(pt.to(do.dtype), do, dv_acc, input_precision="ieee", out_dtype=tl.float32)
        dpt = tl.dot(v, tl.trans(do), input_precision="ieee", out_dtype=tl.float32)
        dst = pt * (dpt - delta[None, :]) * SCALE
        dk_acc = tl.dot(dst.to(q.dtype), q, dk_acc, input_precision="ieee", out_dtype=tl.float32)
        if ADD_DQ is not None:
            dq = tl.dot(tl.trans(dst.to(k.dtype)), k, input_precision="ieee", out_dtype=tl.float32)
            ADD_DQ(DQ, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_H[None, :], dq, MASK_M[:, None] & MASK_H[None, :])
    STORE_DK(DK, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_H[None, :], dk_acc, k_mask)
    STORE_DV(DV, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_HV[None, :], dv_acc, v_mask)


@triton.jit
def dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    SQ,
    SK,
    SHIFT,
    SCALE,
    LOAD_Q: tl.constexpr,
    LOAD_K: tl.constexpr,
    LOAD_V: tl.constexpr,
    LOAD_DO: tl.constexpr,
    LOAD_LSE: tl.constexpr,
    LOAD_DELTA: tl.constexpr,
    STORE_DQ: tl.constexpr,
    SCORE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """dQ of ``BLOCK_SIZE_M`` queries of batch ``BLOCK_ID_B``, summed over the blocks of ``BLOCK_SIZE_N``
    keys they see: ``P = exp(S - lse)``, ``dS = P (dO V^T - delta) SCALE`` and ``dQ += dS K``."""
    BLOCK_ID_M = tl.program_id(axis=0)
    BLOCK_ID_B = tl.program_id(axis=1).to(tl.int64)

    start = BLOCK_ID_M * BLOCK_SIZE_M
    BLOCK_M = (start + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
    BLOCK_H = tl.arange(0, BLOCK_SIZE_H).to(tl.int64)
    BLOCK_HV = tl.arange(0, BLOCK_SIZE_HV).to(tl.int64)

    MASK_M = BLOCK_M < SQ
    MASK_H = BLOCK_H < H
    MASK_HV = BLOCK_HV < HV

    q_mask = MASK_M[:, None] & MASK_H[None, :]
    q = LOAD_Q(Q, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_H[None, :], q_mask)
    do = LOAD_DO(DO, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_HV[None, :], MASK_M[:, None] & MASK_HV[None, :])
    lse = LOAD_LSE(LSE, BLOCK_ID_B, 0, BLOCK_M, MASK_M)
    delta = LOAD_DELTA(DELTA, BLOCK_ID_B, 0, BLOCK_M, MASK_M)
    dq_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_H], tl.float32)
    for k0 in range(0, _keys_end(start, SQ, SK, SHIFT, BLOCK_SIZE_M, CAUSAL), BLOCK_SIZE_N):
        BLOCK_N = (k0 + tl.arange(0, BLOCK_SIZE_N)).to(tl.int64)
        MASK_N = BLOCK_N < SK
        k = LOAD_K(K, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_H[None, :], MASK_N[:, None] & MASK_H[None, :])
        v = LOAD_V(V, BLOCK_ID_B, BLOCK_N[:, None], BLOCK_HV[None, :], MASK_N[:, None] & MASK_HV[None, :])
        s = SCORE(tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32))
        p = tl.where(_seen(BLOCK_M, BLOCK_N, SQ, SK, SHIFT, CAUSAL), tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee", out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * SCALE
        dq_acc = tl.dot(ds.to(k.dtype), k, dq_acc, input_precision="ieee", out_dtype=tl.float32)
    STORE_DQ(DQ, BLOCK_ID_B, BLOCK_M[:, None], BLOCK_H[None, :], dq_acc, q_mask)


KERNELS = {"forward": forward_kernel, "dkdv": dkdv_kernel, "dq": dq_kernel}


def _width(n):
    """Tile width for a head dim of `n`: a power of two, at least 16 (``tl.dot``'s smallest)."""
    return max(16, triton.next_power_of_2(n))


def attention(kernel, tensors, specs, sizes, causal, scale, config, compile=False):
    """Launches flash-attention ``kernel`` (``forward``, ``dkdv`` or ``dq``) on ``tensors`` (in its parameter
    order; ``None`` for an optional one it doesn't read; the forward kernel's past the sixth are its ``EXTRA``) with ``sizes`` (batch, queries, keys, head dims of
    Q/K and V), the ``causal`` diagonal (or ``None``), dS's ``scale`` and ``config`` (blocks, stages, warps);
    ``specs`` are the generated helpers. With ``compile``, returns its ``compiled_launch`` instead of
    launching it."""
    batch, sq, sk, h, hv = sizes
    block_m, block_n, stages, warps = config
    if batch == 0 or sq == 0 or sk == 0:
        return None
    args = dict(
        SQ=sq,
        SK=sk,
        SHIFT=causal or 0,
        **functions(specs),
        H=h,
        HV=hv,
        BLOCK_SIZE_H=_width(h),
        BLOCK_SIZE_HV=_width(hv),
        CAUSAL=causal is not None,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        num_stages=stages,
        num_warps=warps,
    )
    if kernel == "forward":
        args.update(HAS_LSE=tensors[4] is not None, HAS_OFFSET=tensors[5] is not None)
    else:
        args.update(SCALE=scale)
    blocks = triton.cdiv(sk, block_n) if kernel == "dkdv" else triton.cdiv(sq, block_m)
    if kernel == "dkdv" and len(tensors) == 8:
        tensors = [*tensors, None]
    # An optional tensor the kernel doesn't read is passed as the first one, never dereferenced.
    tensors = [t if t is not None else tensors[0] for t in tensors]
    if kernel == "forward":
        # The epilogue's tensors, past the sixth, as one tuple, never empty.
        tensors = [*tensors[:6], tuple(tensors[6:] or tensors[:1])]
    if compile:
        return compiled_launch(KERNELS[kernel], (blocks, batch), tensors, args)
    KERNELS[kernel][(blocks, batch)](*tensors, **args)
