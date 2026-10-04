from lumen.functional.functions import le, where
from lumen.graph import prims
from lumen.graph.tracer import (
    _accum_dtype,
)

__all__ = ["_expand_heads", "_scores"]


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
