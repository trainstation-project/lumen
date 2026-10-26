"""Causal attention of new query tokens against a KV cache."""

import math

from lumen.functional.attention.utils import _expand_heads
from lumen.functional.functions import amax, exp, le, sum, where
from lumen.graph import prims
from lumen.graph.tracer import TracedTensor, _accum_dtype, _common_dtype, _lift, _require_float

__all__ = ["attention_decode"]


def attention_decode(query, key, value, start, scale=None):
    """Attends query tokens to ``key``/``value`` cache rows, query row ``i`` seeing keys ``0..=start + i``.

    Shapes are ``[B, Sq, N, H]``, ``[B, Sk, Nkv, H]``, ``[B, Sk, Nkv, Hv]`` with ``N`` a multiple of ``Nkv``;
    ``start`` is an int or a 0-d integer tensor; returns ``[B, Sq, N, Hv]`` in the query dtype.
    """
    q, k, v, start = _lift(query), _lift(key), _lift(value), _lift(start)

    for name, t in (("query", q), ("key", k), ("value", v)):
        if not isinstance(t, TracedTensor) or t.ndim != 4:
            raise ValueError(f"attention_decode: {name} must be a [B, S, N, H] tensor, got {t!r}")
    if not isinstance(start, int) and not (
        isinstance(start, TracedTensor) and start.ndim == 0 and "int" in start.dtype
    ):
        raise ValueError(f"attention_decode: start must be an int or a 0-d integer tensor, got {start!r}")

    _require_float(q, "attention_decode")
    _common_dtype("attention_decode", (q, k, v))

    b, _, n, h = q.shape
    _, sk, nkv, _ = v.shape

    if k.shape[:3] != (b, sk, nkv) or v.shape[0] != b or k.shape[3] != h or n % nkv:
        raise ValueError(
            "attention_decode: shapes "
            f"{list(q.shape)}, {list(k.shape)}, {list(v.shape)} are not [B, Sq, N, H], [B, Sk, Nkv, H], [B, Sk, Nkv, Hv] "
            "with N a multiple of Nkv"
        )

    if scale is None:
        scale = 1.0 / math.sqrt(h)

    accum = _accum_dtype(q.dtype)
    # Scores are [B, N, Sq, Sk]: contract H, batch over B and the head axis.
    s = prims.dot_general(q, _expand_heads(k, n), (((3,), (3,)), ((0, 2), (0, 2))), accum, accum) * scale
    index = start.dtype if isinstance(start, TracedTensor) else "int32"
    rows = prims.iota(index, s.shape, 2)
    cols = prims.iota(index, s.shape, 3)
    s = where(le(cols, rows + start), s, float("-inf"))
    m = amax(s, -1, keepdim=True)
    e = exp(s - m)
    p = (e / sum(e, -1, keepdim=True)).to(q.dtype)
    # Contract Sk of p [B, N, Sq, Sk] with v [B, Sk, N, Hv], giving [B, N, Sq, Hv].
    o = prims.dot_general(p, _expand_heads(v, n), (((3,), (1,)), ((0, 1), (0, 2))), accum, q.dtype)
    return o.permute(0, 2, 1, 3)
