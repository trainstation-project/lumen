"""Triton flash attention (FlashAttention-2): a forward kernel writing each query row's log-sum-exp, and
backward kernels for dK with dV and for dQ that recompute the probabilities from it. Operands are read and
written through generated helpers that know their layouts (``LOAD_Q(ptr, b, rows, cols, mask)``,
``STORE_O(ptr, b, rows, cols, value, mask)``), and ``SCORE`` applies the traced score ops (casts, scales)."""

import triton
import triton.language as tl

from lumen.codegen.triton import functions


@triton.jit
def _seen(rows, keys, SQ, SK, shift, CAUSAL: tl.constexpr):
    """Which `(query, key)` pairs of the tile count: both in range and, when causal, the key on or before
    the query's diagonal `query + shift`."""
    seen = (rows < SQ)[:, None] & (keys < SK)[None, :]
    if CAUSAL:
        seen = seen & (keys[None, :] <= rows[:, None] + shift)
    return seen


@triton.jit
def _keys_end(start, SQ, SK, shift, BLOCK_M: tl.constexpr, CAUSAL: tl.constexpr):
    """One past the last key the queries of the block at `start` see."""
    end = SK
    if CAUSAL:
        end = tl.minimum(tl.maximum(tl.minimum(start + BLOCK_M, SQ) + shift, 0), SK)
    return end


@triton.jit
def forward_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    OFFSET,
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
    BLOCK_H: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_OFFSET: tl.constexpr,
    HAS_LSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``BLOCK_M`` queries of batch ``program_id(1)``: the keys in blocks of ``BLOCK_N``, an online softmax of
    their scores (running max ``m`` and sum ``l``) and its probabilities, rounded to V's dtype, times V;
    normalized after the last block. A causal block stops at the last key its last query sees (``SHIFT``
    plus ``OFFSET[0]`` past the diagonal)."""
    start = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1).to(tl.int64)
    rows = (start + tl.arange(0, BLOCK_M)).to(tl.int64)
    dh = tl.arange(0, BLOCK_H).to(tl.int64)
    dv = tl.arange(0, BLOCK_HV).to(tl.int64)
    q = LOAD_Q(Q, b, rows[:, None], dh[None, :], (rows < SQ)[:, None] & (dh < H)[None, :])
    shift = SHIFT
    if HAS_OFFSET:
        shift += tl.load(OFFSET).to(tl.int64)
    m = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_HV], tl.float32)
    for k0 in range(0, _keys_end(start, SQ, SK, shift, BLOCK_M, CAUSAL), BLOCK_N):
        keys = (k0 + tl.arange(0, BLOCK_N)).to(tl.int64)
        k = LOAD_K(K, b, keys[:, None], dh[None, :], (keys < SK)[:, None] & (dh < H)[None, :])
        s = SCORE(tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32))
        s = tl.where(_seen(rows, keys, SQ, SK, shift, CAUSAL), s, float("-inf"))
        # Rescale the running sum and output to the new running max; rows that have seen no key keep 0.
        m_new = tl.maximum(m, tl.max(s, 1))
        empty = m_new == float("-inf")
        alpha = tl.where(empty, 1.0, tl.exp(m - m_new))
        p = tl.where(empty[:, None], 0.0, tl.exp(s - m_new[:, None]))
        l = l * alpha + tl.sum(p, 1)
        v = LOAD_V(V, b, keys[:, None], dv[None, :], (keys < SK)[:, None] & (dv < HV)[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision="ieee", out_dtype=tl.float32)
        m = m_new
    if HAS_LSE:
        STORE_LSE(LSE, b, rows, 0, m + tl.log(l), rows < SQ)
    STORE_O(O, b, rows[:, None], dv[None, :], acc / l[:, None], (rows < SQ)[:, None] & (dv < HV)[None, :])


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
    BLOCK_H: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dK and dV of ``BLOCK_N`` keys of batch ``program_id(1)``, summed over the blocks of ``BLOCK_M``
    queries that see them, on transposed scores ``S^T = K Q^T``: ``P^T = exp(S^T - lse)``,
    ``dV += P^T dO``, ``dS^T = P^T (V dO^T - delta) SCALE`` and ``dK += dS^T Q``."""
    start = tl.program_id(0) * BLOCK_N
    b = tl.program_id(1).to(tl.int64)
    keys = (start + tl.arange(0, BLOCK_N)).to(tl.int64)
    dh = tl.arange(0, BLOCK_H).to(tl.int64)
    dv = tl.arange(0, BLOCK_HV).to(tl.int64)
    k_mask = (keys < SK)[:, None] & (dh < H)[None, :]
    v_mask = (keys < SK)[:, None] & (dv < HV)[None, :]
    k = LOAD_K(K, b, keys[:, None], dh[None, :], k_mask)
    v = LOAD_V(V, b, keys[:, None], dv[None, :], v_mask)
    dk_acc = tl.zeros([BLOCK_N, BLOCK_H], tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_HV], tl.float32)
    # A causal key block starts at the first query block that sees its first key.
    first = 0
    if CAUSAL:
        first = tl.maximum(start - SHIFT, 0) // BLOCK_M * BLOCK_M
    for q0 in range(first, SQ, BLOCK_M):
        rows = (q0 + tl.arange(0, BLOCK_M)).to(tl.int64)
        q = LOAD_Q(Q, b, rows[:, None], dh[None, :], (rows < SQ)[:, None] & (dh < H)[None, :])
        do = LOAD_DO(DO, b, rows[:, None], dv[None, :], (rows < SQ)[:, None] & (dv < HV)[None, :])
        lse = LOAD_LSE(LSE, b, 0, rows, rows < SQ)
        delta = LOAD_DELTA(DELTA, b, 0, rows, rows < SQ)
        st = SCORE(tl.dot(k, tl.trans(q), input_precision="ieee", out_dtype=tl.float32))
        seen = (keys < SK)[:, None] & (rows < SQ)[None, :]
        if CAUSAL:
            seen = seen & (keys[:, None] <= rows[None, :] + SHIFT)
        pt = tl.where(seen, tl.exp(st - lse[None, :]), 0.0)
        dv_acc = tl.dot(pt.to(do.dtype), do, dv_acc, input_precision="ieee", out_dtype=tl.float32)
        dpt = tl.dot(v, tl.trans(do), input_precision="ieee", out_dtype=tl.float32)
        dst = pt * (dpt - delta[None, :]) * SCALE
        dk_acc = tl.dot(dst.to(q.dtype), q, dk_acc, input_precision="ieee", out_dtype=tl.float32)
    STORE_DK(DK, b, keys[:, None], dh[None, :], dk_acc, k_mask)
    STORE_DV(DV, b, keys[:, None], dv[None, :], dv_acc, v_mask)


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
    BLOCK_H: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """dQ of ``BLOCK_M`` queries of batch ``program_id(1)``, summed over the blocks of ``BLOCK_N`` keys they
    see: ``P = exp(S - lse)``, ``dS = P (dO V^T - delta) SCALE`` and ``dQ += dS K``."""
    start = tl.program_id(0) * BLOCK_M
    b = tl.program_id(1).to(tl.int64)
    rows = (start + tl.arange(0, BLOCK_M)).to(tl.int64)
    dh = tl.arange(0, BLOCK_H).to(tl.int64)
    dv = tl.arange(0, BLOCK_HV).to(tl.int64)
    q_mask = (rows < SQ)[:, None] & (dh < H)[None, :]
    q = LOAD_Q(Q, b, rows[:, None], dh[None, :], q_mask)
    do = LOAD_DO(DO, b, rows[:, None], dv[None, :], (rows < SQ)[:, None] & (dv < HV)[None, :])
    lse = LOAD_LSE(LSE, b, 0, rows, rows < SQ)
    delta = LOAD_DELTA(DELTA, b, 0, rows, rows < SQ)
    dq_acc = tl.zeros([BLOCK_M, BLOCK_H], tl.float32)
    for k0 in range(0, _keys_end(start, SQ, SK, SHIFT, BLOCK_M, CAUSAL), BLOCK_N):
        keys = (k0 + tl.arange(0, BLOCK_N)).to(tl.int64)
        k = LOAD_K(K, b, keys[:, None], dh[None, :], (keys < SK)[:, None] & (dh < H)[None, :])
        v = LOAD_V(V, b, keys[:, None], dv[None, :], (keys < SK)[:, None] & (dv < HV)[None, :])
        s = SCORE(tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32))
        p = tl.where(_seen(rows, keys, SQ, SK, SHIFT, CAUSAL), tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee", out_dtype=tl.float32)
        ds = p * (dp - delta[:, None]) * SCALE
        dq_acc = tl.dot(ds.to(k.dtype), k, dq_acc, input_precision="ieee", out_dtype=tl.float32)
    STORE_DQ(DQ, b, rows[:, None], dh[None, :], dq_acc, q_mask)


KERNELS = {"forward": forward_kernel, "dkdv": dkdv_kernel, "dq": dq_kernel}


def _width(n):
    """Tile width for a head dim of `n`: a power of two, at least 16 (``tl.dot``'s smallest)."""
    return max(16, triton.next_power_of_2(n))


def attention(kernel, tensors, specs, sizes, causal, scale, config):
    """Launches flash-attention ``kernel`` (``forward``, ``dkdv`` or ``dq``) on ``tensors`` (in its parameter
    order; ``None`` for an optional one it doesn't read) with ``sizes`` (batch, queries, keys, head dims of
    Q/K and V), the ``causal`` diagonal (or ``None``), dS's ``scale`` and ``config`` (blocks, stages, warps);
    ``specs`` are the generated helpers."""
    batch, sq, sk, h, hv = sizes
    block_m, block_n, stages, warps = config
    if batch == 0 or sq == 0 or sk == 0:
        return
    args = dict(
        SQ=sq,
        SK=sk,
        SHIFT=causal or 0,
        **functions(specs),
        H=h,
        HV=hv,
        BLOCK_H=_width(h),
        BLOCK_HV=_width(hv),
        CAUSAL=causal is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_stages=stages,
        num_warps=warps,
    )
    if kernel == "forward":
        args.update(HAS_LSE=tensors[4] is not None, HAS_OFFSET=tensors[5] is not None)
    else:
        args.update(SCALE=scale)
    blocks = triton.cdiv(sk, block_n) if kernel == "dkdv" else triton.cdiv(sq, block_m)
    # An optional tensor the kernel doesn't read is passed as the first one, never dereferenced.
    tensors = [t if t is not None else tensors[0] for t in tensors]
    KERNELS[kernel][(blocks, batch)](*tensors, **args)
