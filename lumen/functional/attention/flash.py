import math

from lumen.autograd.function import Function
from lumen.functional.attention.utils import _expand_heads, _scores
from lumen.functional.functions import amax, exp, le, log, sum, where
from lumen.graph import prims
from lumen.graph.tracer import (
    TracedTensor,
    _accum_dtype,
    _common_dtype,
    _lift,
    _require_float,
)

__all__ = ["flash_attention"]


class _FlashAttention(Function):
    """Attention with FlashAttention-2's backward (MLX's
    ``ScaledDotProductAttentionVJP``): the forward also gives each query
    row's log-sum-exp of its scores, from which the backward recomputes the
    probabilities, never saving them."""

    @staticmethod
    def forward(ctx, q, k, v, *, scale, causal):
        n = q.shape[2]
        accum = _accum_dtype(q.dtype)
        s = _scores(q, _expand_heads(k, n), scale, causal)
        # The softmax, its max and sum kept for the log-sum-exp.
        m = amax(s, -1, keepdim=True)
        e = exp(s - m)
        total = sum(e, -1, keepdim=True)
        p = (e / total).to(q.dtype)
        # [B, N, Sq, Hv], then [B, Sq, N, Hv].
        o = prims.dot_general(p, _expand_heads(v, n), (((3,), (1,)), ((0, 1), (0, 2))), accum, q.dtype)
        o = o.permute(0, 2, 1, 3)
        ctx.scale, ctx.causal = scale, causal
        ctx.save_for_backward(q, k, v, o, lse=m + log(total))
        return o

    @staticmethod
    def backward(ctx, do):
        # On the scores transposed, S^T = K Q^T [B, N, Sk, Sq] (a row a key,
        # as dV = P^T dO and dK = dS^T Q read them): not the forward's Q K^T,
        # so recomputed, never its [Sq, Sk] values read back.
        q, k, v, o = ctx.saved_tensors
        lse = ctx.saved_named_tensors["lse"]
        b, sq, n, _ = q.shape
        sk, nkv = k.shape[1], k.shape[2]
        dtype, accum = q.dtype, _accum_dtype(q.dtype)
        ke, ve = _expand_heads(k, n), _expand_heads(v, n)
        st = prims.dot_general(ke, q, (((3,), (3,)), ((0, 2), (0, 2))), accum, accum) * ctx.scale
        if ctx.causal:
            keys = prims.iota("int64", st.shape, 2)
            queries = prims.iota("int64", st.shape, 3)
            st = where(le(keys, queries + (sk - sq)), st, float("-inf"))
        # P^T, from each query's log-sum-exp; D = rowsum(dO * O), per query.
        pt = exp(st - lse.reshape(b, n, 1, sq))
        # A dot over the head dimension, accumulated in float32: no value
        # rounded (O by p @ v, dO by its producer) is widened.
        d = prims.dot_general(do, o, (((3,), (3,)), ((0, 1, 2), (0, 1, 2))), accum, accum)
        d = d.permute(0, 2, 1).reshape(b, n, 1, sq)
        # dP^T = V dO^T; dS^T = P^T (dP^T - D) * scale.
        dpt = prims.dot_general(ve, do, (((3,), (3,)), ((0, 2), (0, 2))), accum, accum)
        dst = (pt * (dpt - d) * ctx.scale).to(dtype)
        pt = pt.to(dtype)
        # [B, N, Sk, Hv], [B, N, Sk, H] and [B, N, Sq, H]: in float32.
        dv = prims.dot_general(pt, do, (((3,), (1,)), ((0, 1), (0, 2))), accum, accum)
        dk = prims.dot_general(dst, q, (((3,), (1,)), ((0, 1), (0, 2))), accum, accum)
        dq = prims.dot_general(dst, ke, (((2,), (1,)), ((0, 1), (0, 2))), accum, accum)
        dq, dk, dv = (t.permute(0, 2, 1, 3) for t in (dq, dk, dv))
        if nkv != n:
            # Summed over each key and value head's group of query heads.
            dk = sum(dk.reshape(b, sk, nkv, n // nkv, k.shape[3]), 3)
            dv = sum(dv.reshape(b, sk, nkv, n // nkv, v.shape[3]), 3)
        return dq.to(dtype), dk.to(dtype), dv.to(dtype)


def flash_attention(query, key, value, scale=None, is_causal=False):
    """``softmax(query @ key^T * scale) @ value`` over each batch and head,
    as flash-attn lays them out: ``query`` ``[B, Sq, N, H]`` (batch,
    sequence, heads, head dim), ``key`` ``[B, Sk, Nkv, H]`` and ``value``
    ``[B, Sk, Nkv, Hv]``, ``N`` a multiple of ``Nkv`` (grouped-query
    attention: each key and value head serves ``N / Nkv`` query heads);
    the result ``[B, Sq, N, Hv]``. ``scale`` defaults to ``1 / sqrt(H)``.
    ``is_causal`` masks out key ``j`` for query ``i`` where
    ``j > i + Sk - Sq`` (aligned to the bottom right, as flash-attn and
    MLX: with a KV cache, each new query sees every earlier key).

    Traced as its primitives: the scores (accumulated, and the softmax
    computed, in float32 or wider), the softmax, then the probabilities in
    the inputs' dtype times ``value``. The MPS compiler runs them as one
    flash-attention kernel (``lumen.config.compiler.flash_attention``), as
    it does attention written out. An ``autograd.Function`` (``_FlashAttention``):
    its gradient is FlashAttention-2's, the probabilities recomputed from
    each row's log-sum-exp, not saved; on MPS, two more kernels."""

    q = _lift(query)
    k = _lift(key)
    v = _lift(value)

    for name, t in (("query", q), ("key", k), ("value", v)):
        if not isinstance(t, TracedTensor) or t.ndim != 4:
            raise ValueError(f"flash_attention: {name} must be a [B, S, N, H] tensor, got {t!r}")

    _require_float(q, "flash_attention")
    _common_dtype("flash_attention", (q, k, v))

    b, _, n, h = q.shape
    _, sk, nkv, _ = v.shape

    if k.shape[:3] != (b, sk, nkv) or v.shape[0] != b or k.shape[3] != h or n % nkv:
        raise ValueError(
            "flash_attention: shapes "
            f"{list(q.shape)}, {list(k.shape)}, {list(v.shape)} are not [B, Sq, N, H], [B, Sk, Nkv, H], [B, Sk, Nkv, Hv] "
            "with N a multiple of Nkv"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(h)

    return _FlashAttention.apply(q, k, v, scale=scale, causal=is_causal)
