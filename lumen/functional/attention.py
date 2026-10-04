import math

from lumen.functional.functions import le, softmax, where
from lumen.graph import prims
from lumen.graph.tracer import (
    TracedTensor,
    _accum_dtype,
    _common_dtype,
    _lift,
    _require_float,
)


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
    it does attention written out. Its gradient, as any attention's, is
    FlashAttention-2's (``lumen/autograd/attention.py``): on MPS, two more
    kernels."""

    q = _lift(query)
    k = _lift(key)
    v = _lift(value)

    for name, t in (("query", q), ("key", k), ("value", v)):
        if not isinstance(t, TracedTensor) or t.ndim != 4:
            raise ValueError(f"flash_attention: {name} must be a [B, S, N, H] tensor, got {t!r}")

    _require_float(q, "flash_attention")
    _common_dtype("flash_attention", (q, k, v))

    b, sq, n, h = q.shape
    _, sk, nkv, hv = v.shape

    if k.shape[:3] != (b, sk, nkv) or v.shape[0] != b or k.shape[3] != h or n % nkv:
        raise ValueError(
            "flash_attention: shapes "
            f"{list(q.shape)}, {list(k.shape)}, {list(v.shape)} are not [B, Sq, N, H], [B, Sk, Nkv, H], [B, Sk, Nkv, Hv] "
            "with N a multiple of Nkv"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(h)
    accum = _accum_dtype(q.dtype)
    s = _scores(q, _expand_heads(k, n), scale, is_causal)
    p = softmax(s, -1).to(q.dtype)
    # [B, N, Sq, Hv], then [B, Sq, N, Hv].
    o = prims.dot_general(p, _expand_heads(v, n), (((3,), (1,)), ((0, 1), (0, 2))), accum, q.dtype)
    return o.permute(0, 2, 1, 3)


def _expand_heads(x, n):
    """``x`` ``[B, S, Nkv, H]`` with each head repeated for its group of
    ``n / Nkv`` query heads: ``[B, S, n, H]``."""
    b, s, nkv, h = x.shape
    return x if nkv == n else x.unsqueeze(3).expand(b, s, nkv, n // nkv, h).reshape(b, s, n, h)


def _scores(q, k, scale, causal):
    """``q k^T * scale`` ``[B, N, Sq, Sk]`` in the accumulation dtype, the
    keys a causal mask hides ``-inf``."""
    accum = _accum_dtype(q.dtype)
    s = prims.dot_general(q, k, (((3,), (3,)), ((0, 2), (0, 2))), accum, accum) * scale
    if causal:
        sq, sk = s.shape[2], s.shape[3]
        rows = prims.iota("int64", s.shape, 2)
        cols = prims.iota("int64", s.shape, 3)
        s = where(le(cols, rows + (sk - sq)), s, float("-inf"))
    return s


def naive_attention(query, key, value, scale=None, is_causal=False):
    """:func:`flash_attention` written out as it was first
    (``main``'s): ``softmax(query @ key^T * scale) @ value`` over each batch and head,
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
    it does attention written out."""

    q = _lift(query)
    k = _lift(key)
    v = _lift(value)

    for name, t in (("query", q), ("key", k), ("value", v)):
        if not isinstance(t, TracedTensor) or t.ndim != 4:
            raise ValueError(f"naive_attention: {name} must be a [B, S, N, H] tensor, got {t!r}")

    _require_float(q, "naive_attention")
    _common_dtype("naive_attention", (q, k, v))

    b, sq, n, h = q.shape
    _, sk, nkv, hv = v.shape

    if k.shape[:3] != (b, sk, nkv) or v.shape[0] != b or k.shape[3] != h or n % nkv:
        raise ValueError(
            "naive_attention: shapes "
            f"{list(q.shape)}, {list(k.shape)}, {list(v.shape)} are not [B, Sq, N, H], [B, Sk, Nkv, H], [B, Sk, Nkv, Hv] "
            "with N a multiple of Nkv"
        )

    if nkv != n:
        # Each key and value head, for its group of query heads.
        g = n // nkv
        k = k.unsqueeze(3).expand(b, sk, nkv, g, h).reshape(b, sk, n, h)
        v = v.unsqueeze(3).expand(b, sk, nkv, g, hv).reshape(b, sk, n, hv)

    accum = _accum_dtype(q.dtype)
    if scale is None:
        scale = 1.0 / math.sqrt(h)

    # [B, N, Sq, Sk], in the accumulation dtype.
    s = prims.dot_general(q, k, (((3,), (3,)), ((0, 2), (0, 2))), accum, accum)
    s = s * scale

    if is_causal:
        rows = prims.iota("int64", s.shape, 2)
        cols = prims.iota("int64", s.shape, 3)
        s = where(le(cols, rows + (sk - sq)), s, float("-inf"))

    p = softmax(s, -1).to(q.dtype)
    # [B, N, Sq, Hv], then [B, Sq, N, Hv].
    o = prims.dot_general(p, v, (((3,), (1,)), ((0, 1), (0, 2))), accum, q.dtype)

    return o.permute(0, 2, 1, 3)
