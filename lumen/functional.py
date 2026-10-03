"""The ops, as functions: ``import lumen.functional as F`` (PyTorch's
``torch`` and ``torch.nn.functional`` functions, by the same names).

Each records primitives (``lumen.graph.prims``) into the graph being traced
by ``lumen.compile``, so each takes traced tensors (and Python scalars).
Tensors keep only their operators (``+ - * / @ == < []``, ``-x``), their
layout (``reshape``, ``permute``, ``transpose``, ...) and dtype casts
(``to``, ``float``, ...); every other op is here.

No op changes a dtype the program does not ask for: operands share a dtype
(a Python scalar takes its tensor operand's, and must be of its kind),
floating-point functions take floating-point tensors, and sums and means
reduce in their input's dtype. Convert first with ``.to(dtype)``
(``F.sum(x.float())``).
"""

import math

from lumen.graph import prims
from lumen.graph.tracer import (
    TracedTensor,
    _accum_dtype,
    _as_tensor,
    _broadcast_shapes,
    _broadcast_to,
    _common_dtype,
    _dims,
    _elementwise,
    _is_float,
    _le,
    _lift,
    _min,
    _ne,
    _require_float,
    _true_div,
)

__all__ = [
    # arithmetic and comparison
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "maximum",
    "minimum",
    "where",
    # elementwise functions
    "exp",
    "log",
    "sqrt",
    "tanh",
    "sigmoid",
    "relu",
    # reductions
    "sum",
    "mean",
    "amax",
    "max",
    # normalizations
    "softmax",
    "log_softmax",
    "rms_norm",
    # contractions
    "matmul",
    "scaled_dot_product_attention",
]


# ---------------------------------------------------------------------
# arithmetic and comparison
# ---------------------------------------------------------------------


def add(input, other):
    return _elementwise(prims.add, input, other)


def sub(input, other):
    return _elementwise(prims.sub, input, other)


def mul(input, other):
    return _elementwise(prims.mul, input, other)


def div(input, other):
    """True division: floating-point operands."""
    return _true_div(input, other)


def neg(input):
    return prims.neg(_lift(input))


def eq(input, other):
    return _elementwise(prims.eq, input, other)


def ne(input, other):
    return _elementwise(_ne, input, other)


def lt(input, other):
    return _elementwise(prims.lt, input, other)


def le(input, other):
    return _elementwise(_le, input, other)


def gt(input, other):
    return _elementwise(prims.lt, other, input)


def ge(input, other):
    return _elementwise(_le, other, input)


def maximum(input, other):
    return _elementwise(prims.max, input, other)


def minimum(input, other):
    return _elementwise(_min, input, other)


def where(condition, input, other):
    """``input`` where ``condition`` (a bool tensor), else ``other``,
    broadcast together."""
    condition, input, other = _lift(condition), _lift(input), _lift(other)
    if not isinstance(condition, TracedTensor) or condition.dtype != "bool":
        raise TypeError("where expected condition to be a bool tensor")
    dtype = _common_dtype("where", (input, other))
    input, other = _as_tensor(input, dtype, "where"), _as_tensor(other, dtype, "where")
    shape = _broadcast_shapes(condition.shape, input.shape, other.shape)
    return prims.select(*(_broadcast_to(t, shape) for t in (condition, input, other)))


# ---------------------------------------------------------------------
# elementwise functions
# ---------------------------------------------------------------------


def exp(input):
    return prims.exp(_require_float(_lift(input), "exp"))


def log(input):
    return prims.log(_require_float(_lift(input), "log"))


def sqrt(input):
    return prims.sqrt(_require_float(_lift(input), "sqrt"))


def tanh(input):
    return prims.tanh(_require_float(_lift(input), "tanh"))


def sigmoid(input):
    return prims.logistic(_require_float(_lift(input), "sigmoid"))


def relu(input):
    return _elementwise(prims.max, input, 0)


# ---------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------


def _keep(input, out, dims, keepdim):
    """Reduction ``out`` of ``input`` over ``dims``, with them kept (as
    size 1) if ``keepdim``."""
    if not keepdim:
        return out
    return out.reshape(tuple(1 if d in dims else n for d, n in enumerate(input.shape)))


def sum(input, dim=None, keepdim=False):
    """Sum over ``dim`` (all dimensions if None), accumulated in the
    tensor's dtype, the result's (integers wrap): for another, convert
    first (``F.sum(x.float())``); a cast of the result after it runs in
    the reduction's kernel."""
    dims = _dims(dim, input.ndim)
    return _keep(input, prims.reduce_sum(input, dims, input.dtype), dims, keepdim)


def mean(input, dim=None, keepdim=False):
    """Mean over ``dim`` (all dimensions if None), in the tensor's dtype, as
    :func:`sum`."""
    if not _is_float(input.dtype):
        raise RuntimeError(f"mean(): input dtype must be floating point, got {input.dtype}")
    dims = _dims(dim, input.ndim)
    count = math.prod(input.shape[d] for d in dims)
    return sum(input, dims, keepdim) / count


def amax(input, dim=(), keepdim=False):
    """Maximum over ``dim`` (all dimensions if empty)."""
    dims = _dims(dim, input.ndim)
    return _keep(input, prims.reduce_max(input, dims), dims, keepdim)


def max(input, other=None):
    """``max(input)`` over every element, or the elementwise
    ``max(input, other)``. ``max(input, dim)``, which also returns indices,
    is not supported: use :func:`amax`."""
    if other is None:
        return amax(input)
    other = _lift(other)
    if isinstance(other, TracedTensor):
        return maximum(input, other)
    raise NotImplementedError("max(input, dim) returns indices, which lumen does not support yet; use amax")


# ---------------------------------------------------------------------
# normalizations
# ---------------------------------------------------------------------


def softmax(input, dim):
    """``exp(x - max) / sum(exp(x - max))`` along ``dim``, traced as those
    primitives: over the last dimension, the MPS compiler runs them as one
    row kernel (a chain of normalization diamonds,
    ``compiler/mps/diamonds.rs``)."""
    x = _require_float(_lift(input), "softmax")
    e = exp(x - amax(x, dim, keepdim=True))
    return e / sum(e, dim, keepdim=True)


def log_softmax(input, dim):
    x = _require_float(_lift(input), "log_softmax")
    shifted = x - amax(x, dim, keepdim=True)
    return shifted - log(sum(exp(shifted), dim, keepdim=True))


def rms_norm(input, normalized_shape, weight=None, eps=None):
    """``torch.nn.functional.rms_norm``: ``input`` normalized by its root mean
    square over the last dimension (``normalized_shape``, its size),
    ``input / sqrt(mean(input^2) + eps)``, times ``weight`` if given. ``eps``
    defaults to the dtype's machine epsilon, as in torch. Traced as its
    primitives, which the MPS compiler runs as one row kernel (a
    normalization diamond), with the ops computing ``input`` fused in."""
    x = _require_float(_lift(input), "rms_norm")
    shape = [normalized_shape] if isinstance(normalized_shape, int) else list(normalized_shape)

    if shape != list(x.shape[-1:]):
        raise NotImplementedError(f"rms_norm normalizes the last dimension, of size {x.shape[-1:]}, got {shape}")

    weight = _lift(weight)
    if weight is not None and (weight.dtype != x.dtype or list(weight.shape) != shape):
        raise TypeError(f"rms_norm: weight must be {x.dtype}{shape}, got {weight.dtype}{list(weight.shape)}")

    if eps is None:
        eps = {"float16": 2.0**-10, "bfloat16": 2.0**-7, "float64": 2.0**-52}.get(x.dtype, 2.0**-23)

    # Normalized in the accumulation dtype (float32 for narrower floats),
    # then cast back and scaled by the weight in x's dtype.
    h = x.to(_accum_dtype(x.dtype))
    y = (h / sqrt(mean(h * h, -1, keepdim=True) + eps)).to(x.dtype)

    if weight is not None:
        y = y * weight

    return y


# ---------------------------------------------------------------------
# contractions
# ---------------------------------------------------------------------


def matmul(input, other):
    """``input @ other`` with torch's rules: 1-d operands are vectors, and
    the dimensions before the last two are batch dimensions, broadcast. It
    accumulates floats in float32 (float64 in float64); its result is of
    the inputs' dtype."""
    input, other = _lift(input), _lift(other)
    _common_dtype("matmul", (input, other))
    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D")
    x = input.unsqueeze(0) if input.ndim == 1 else input
    y = other.unsqueeze(-1) if other.ndim == 1 else other
    batch = _broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x, y = _broadcast_to(x, batch + x.shape[-2:]), _broadcast_to(y, batch + y.shape[-2:])
    b = tuple(range(len(batch)))
    # Accumulated in float32 (or wider), the result in the inputs' dtype.
    dims = (((len(b) + 1,), (len(b),)), (b, b))
    out = prims.dot_general(x, y, dims, _accum_dtype(x.dtype), x.dtype)
    if input.ndim == 1:
        out = out.squeeze(-2)
    if other.ndim == 1:
        out = out.squeeze(-1)
    return out


def scaled_dot_product_attention(query, key, value, scale=None, is_causal=False):
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
    it does attention written out."""
    q, k, v = _lift(query), _lift(key), _lift(value)
    for name, t in (("query", q), ("key", k), ("value", v)):
        if not isinstance(t, TracedTensor) or t.ndim != 4:
            raise ValueError(f"scaled_dot_product_attention: {name} must be a [B, S, N, H] tensor, got {t!r}")
    _require_float(q, "scaled_dot_product_attention")
    _common_dtype("scaled_dot_product_attention", (q, k, v))
    (b, sq, n, h), (_, sk, nkv, hv) = q.shape, v.shape
    if k.shape[:3] != (b, sk, nkv) or v.shape[0] != b or k.shape[3] != h or n % nkv:
        raise ValueError(
            "scaled_dot_product_attention: shapes "
            f"{list(q.shape)}, {list(k.shape)}, {list(v.shape)} are not [B, Sq, N, H], [B, Sk, Nkv, H], [B, Sk, Nkv, Hv] "
            "with N a multiple of Nkv"
        )
    if nkv != n:
        # Each key and value head, for its group of query heads.
        g = n // nkv
        k = k.unsqueeze(3).expand(b, sk, nkv, g, h).reshape(b, sk, n, h)
        v = v.unsqueeze(3).expand(b, sk, nkv, g, hv).reshape(b, sk, n, hv)
    accum = _accum_dtype(q.dtype)
    # [B, N, Sq, Sk], in the accumulation dtype.
    s = prims.dot_general(q, k, (((3,), (3,)), ((0, 2), (0, 2))), accum, accum)
    s = s * (1.0 / math.sqrt(h) if scale is None else scale)
    if is_causal:
        rows = prims.iota("int64", s.shape, 2)
        cols = prims.iota("int64", s.shape, 3)
        s = where(le(cols, rows + (sk - sq)), s, float("-inf"))
    p = softmax(s, -1).to(q.dtype)
    # [B, N, Sq, Hv], then [B, Sq, N, Hv].
    o = prims.dot_general(p, v, (((3,), (1,)), ((0, 1), (0, 2))), accum, q.dtype)
    return o.permute(0, 2, 1, 3)
