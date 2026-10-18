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

import builtins
import math

from lumen.autograd.function import Function
from lumen.graph import prims
from lumen.graph.tracer import (
    TracedTensor,
    _accum_dtype,
    _as_tensor,
    _broadcast_shapes,
    _broadcast_to,
    _common_dtype,
    _dim,
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
    # random
    "dropout",
    # reductions
    "sum",
    "mean",
    "amax",
    "max",
    # scans
    "cumsum",
    # normalizations
    "softmax",
    "log_softmax",
    "rms_norm",
    # contractions
    "matmul",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "asinh",
    "acosh",
    "atanh",
    "floor",
    "ceil",
    "round",
    "rint",
    "log1p",
    "expm1",
    "exp2",
    "log2",
    "cbrt",
    "erfinv",
    "abs",
    "sign",
    "square",
    "reciprocal",
    "pow",
    "nextafter",
    "isfinite",
    "isnan",
    "logical_and",
    "logical_or",
    "logical_not",
    "clamp",
    "amin",
    "min",
    "any",
    "all",
    "argmax",
    "argmin",
    "cat",
    "stack",
    "triu",
    "diagonal",
    "pad",
    "gather",
    "einsum",
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
    """True division: floating-point operands. By a scalar (a Python
    number, a runtime scalar), traced as a product with its reciprocal."""
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
    if not isinstance(input, TracedTensor) and not isinstance(other, TracedTensor):
        # Two Python scalars: of their kind's default dtype (torch's).
        kind = "bool" if isinstance(input, bool) and isinstance(other, bool) else None
        ints = builtins.all(isinstance(v, int) for v in (input, other))
        dtype = kind or ("int64" if ints else "float32")
        input = prims.full(condition.shape, input, dtype)
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


def dropout(input, p=0.5, training=True):
    """Each element zeroed with probability ``p``, the others scaled by
    ``1 / (1 - p)`` (``F.dropout``); ``input`` as is unless ``training``.
    An element is dropped where its random bits (``lumen.random``, the
    generator's next ``input.numel`` numbers) are below ``p * 2**32``: the
    mask is computed in the kernels applying it, forward and backward,
    never stored."""
    input = _require_float(_lift(input), "dropout")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
    if not training or p == 0.0:
        return input
    if builtins.round(p * 2**32) >= 2**32:
        return where(prims.full(tuple(input.shape), True, "bool"), 0.0, input)
    return where(_dropped(input.shape, p), 0.0, input * (1.0 / (1.0 - p)))


def _dropped(shape, p):
    """Which elements of ``shape`` dropout with probability ``p`` (below 1)
    drops: those whose random bits (the generator's next ``numel``) are
    below ``p * 2**32``."""
    from lumen import random

    bits = random._bits(shape)
    return bits < prims.full(tuple(shape), builtins.round(p * 2**32), "uint32")


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
# scans
# ---------------------------------------------------------------------


def cumsum(input, dim, dtype=None):
    """The cumulative sum along ``dim`` (``torch.cumsum``), accumulated in
    ``dtype``, the result's: the tensor's if None (integers wrap), or
    float32 for a float16 or bfloat16 tensor, each element widened exactly
    as its kernel reads it; for any other, the tensor converted first."""
    x = _lift(input)
    dtype = x.dtype if dtype is None else dtype
    widened = dtype == "float32" and x.dtype in ("float16", "bfloat16")
    if dtype != x.dtype and not widened:
        x = x.to(dtype=dtype)
    return prims.cumsum(x, _dim(dim, x.ndim), False, dtype)


# ---------------------------------------------------------------------
# normalizations
# ---------------------------------------------------------------------


def softmax(input, dim):
    """``exp(x - max) / sum(exp(x - max))`` along ``dim``, traced as those
    primitives: over the last dimension, the MPS compiler runs them as one
    row kernel (a chain of normalization diamonds,
    ``graph_compiler/mps/diamonds.rs``)."""
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


def matmul(input, other, accum_dtype, output_dtype):
    """``input @ other`` with torch's rules: 1-d operands are vectors, and
    the dimensions before the last two are batch dimensions, broadcast. It
    accumulates in ``accum_dtype``, its result of ``output_dtype``. ``@``
    infers them: floats accumulate in float32 (float64 in float64), the
    result of the inputs' dtype."""

    input = _lift(input)
    other = _lift(other)

    _common_dtype("matmul", (input, other))

    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D")

    # A batch of rows times a matrix (``x @ w``, a weight): the batch folded
    # into the rows, one matmul reading the matrix in place (torch's fold),
    # not the matrix broadcast over the batch.
    if input.ndim > 2 and other.ndim == 2:
        k = input.shape[-1]
        rows = input.reshape(math.prod(input.shape[:-1]), k)
        out = prims.dot_general(rows, other, (((1,), (0,)), ((), ())), accum_dtype, output_dtype)
        return out.reshape(*input.shape[:-1], other.shape[-1])

    x = input.unsqueeze(0) if input.ndim == 1 else input
    y = other.unsqueeze(-1) if other.ndim == 1 else other

    batch = _broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x, y = _broadcast_to(x, batch + x.shape[-2:]), _broadcast_to(y, batch + y.shape[-2:])

    b = tuple(range(len(batch)))
    dims = (((len(b) + 1,), (len(b),)), (b, b))

    out = prims.dot_general(x, y, dims, accum_dtype, output_dtype)
    if input.ndim == 1:
        out = out.squeeze(-2)

    if other.ndim == 1:
        out = out.squeeze(-1)

    return out


# -- elementwise functions beyond the core ones ---------------------------


def _float_fn(prim, name):
    def f(input):
        return prim(_require_float(_lift(input), name))

    f.__name__ = name
    f.__doc__ = f"Elementwise ``{name}`` of a floating-point tensor."
    return f


sin = _float_fn(prims.sin, "sin")
cos = _float_fn(prims.cos, "cos")
tan = _float_fn(prims.tan, "tan")
asin = _float_fn(prims.asin, "asin")
acos = _float_fn(prims.acos, "acos")
atan = _float_fn(prims.atan, "atan")
sinh = _float_fn(prims.sinh, "sinh")
cosh = _float_fn(prims.cosh, "cosh")
asinh = _float_fn(prims.asinh, "asinh")
acosh = _float_fn(prims.acosh, "acosh")
atanh = _float_fn(prims.atanh, "atanh")


def _integral_fn(prim, name):
    """``prim`` of a float; an integer as it is (already integral)."""

    def f(input):
        input = _lift(input)
        return input if not _is_float(input.dtype) else prim(input)

    f.__name__ = name
    f.__doc__ = f"Elementwise ``{name}``: an integer tensor as it is."
    return f


floor = _integral_fn(prims.floor, "floor")
ceil = _integral_fn(prims.ceil, "ceil")
# To the nearest integer, ties to even (``torch.round``, ``numpy.rint``).
round = _float_fn(prims.round, "round")
rint = _float_fn(prims.round, "rint")
log1p = _float_fn(prims.log1p, "log1p")
expm1 = _float_fn(prims.expm1, "expm1")
exp2 = _float_fn(prims.exp2, "exp2")
log2 = _float_fn(prims.log2, "log2")
cbrt = _float_fn(prims.cbrt, "cbrt")
erfinv = _float_fn(prims.erf_inv, "erfinv")


def abs(input):
    return prims.abs(_lift(input))


def sign(input):
    """-1, 0 or 1 by ``input``'s sign (a float's zero and NaN as they are)."""
    return prims.sign(_lift(input))


def square(input):
    input = _lift(input)
    return prims.mul(input, input)


def reciprocal(input):
    return _true_div(1.0, _lift(input))


def pow(input, exponent):
    """``input ** exponent`` (an integer's to a non-negative int exponent)."""
    return _lift(input) ** exponent


def nextafter(input, other):
    """The float after ``input`` towards ``other``, elementwise."""
    return _elementwise(prims.nextafter, input, other)


def isfinite(input):
    input = _lift(input)
    if not _is_float(input.dtype):
        return prims.full(input.shape, True, "bool")
    return prims.is_finite(input)


def isnan(input):
    input = _lift(input)
    if not _is_float(input.dtype):
        return prims.full(input.shape, False, "bool")
    return _ne(input, input)


def _truth(input):
    """``input`` as bools: nonzero (a Python scalar's truth)."""
    input = _lift(input)
    if not isinstance(input, TracedTensor):
        return bool(input)
    return input if input.dtype == "bool" else _ne(input, prims.full(input.shape, 0, input.dtype))


def logical_and(input, other):
    return _elementwise(prims.and_, _truth(input), _truth(other))


def logical_or(input, other):
    return _elementwise(prims.or_, _truth(input), _truth(other))


def logical_not(input):
    return prims.not_(_truth(input))


def clamp(input, min=None, max=None):
    """``input`` clamped to ``[min, max]`` (either may be None, a scalar or a
    tensor)."""
    out = _lift(input)
    if min is not None:
        out = maximum(out, min)
    if max is not None:
        out = minimum(out, max)
    return out


# -- reductions -------------------------------------------------------------


def _flip(input):
    """An order-reversing bijection of ``input``'s values: its negation, or
    its bitwise not (unsigned integers, bools)."""
    if input.dtype == "bool" or input.dtype.startswith("uint"):
        return prims.not_(input)
    return prims.neg(input)


def amin(input, dim=(), keepdim=False):
    """Minimum over ``dim`` (all dimensions if empty)."""
    input = _lift(input)
    return _flip(amax(_flip(input), dim, keepdim))


def min(input, other=None):
    """``min(input)`` over every element, or the elementwise
    ``min(input, other)``."""
    if other is None:
        return amin(input)
    return minimum(input, other)


def any(input, dim=None, keepdim=False):
    """Whether any element over ``dim`` (all if None) is nonzero."""
    return amax(_truth(input), () if dim is None else dim, keepdim)


def all(input, dim=None, keepdim=False):
    """Whether every element over ``dim`` (all if None) is nonzero."""
    return logical_not(any(logical_not(input), dim, keepdim))


def _arg(input, dim, keepdim, best):
    input = _lift(input)
    if dim is None:
        flat = input.reshape((-1,))
        return _arg(flat, 0, False, best)
    d = _dim(dim, input.ndim)
    target = best(input, d, True)
    index = prims.iota("int32", input.shape, d)
    big = prims.full(input.shape, input.shape[d], "int32")
    hit = prims.eq(input, _broadcast_to(target, input.shape))
    return amin(prims.select(hit, index, big), d, keepdim)


def argmax(input, dim=None, keepdim=False):
    """The index of the (first) maximum over ``dim`` (of every element, if
    None), as int32."""
    return _arg(input, dim, keepdim, amax)


def argmin(input, dim=None, keepdim=False):
    """The index of the (first) minimum over ``dim`` (of every element, if
    None), as int32."""
    return _arg(input, dim, keepdim, amin)


# -- layout -------------------------------------------------------------------


def cat(tensors, dim=0):
    """``tensors`` one after another along ``dim`` (``torch.cat``)."""
    tensors = [_lift(t) for t in tensors]
    return prims.concatenate(tensors, _dim(dim, tensors[0].ndim))


def stack(tensors, dim=0):
    """``tensors`` (of one shape) stacked along a new dimension ``dim``."""
    tensors = [_lift(t) for t in tensors]
    d = _dim(dim, tensors[0].ndim + 1)
    return cat([t.unsqueeze(d) for t in tensors], d)


def triu(input, diagonal=0):
    """``input``'s upper triangle (its last two dimensions): the elements on
    and above diagonal ``diagonal``, the others zero."""
    input = _lift(input)
    r = input.ndim
    rows = prims.iota("int32", input.shape, r - 2)
    cols = prims.iota("int32", input.shape, r - 1)
    keep = _le(prims.add(rows, prims.full(input.shape, diagonal, "int32")), cols)
    return prims.select(keep, input, prims.full(input.shape, 0, input.dtype))


def diagonal(input, offset=0, dim1=0, dim2=1):
    """The diagonal of ``input``'s dimensions ``dim1`` and ``dim2`` (two
    dimensions), ``offset`` above the main one."""
    input = _lift(input)
    if input.ndim != 2 or (_dim(dim1, 2), _dim(dim2, 2)) != (0, 1):
        raise NotImplementedError("diagonal takes a matrix's (dim1=0, dim2=1)")
    n, m = input.shape
    length = builtins.max(0, builtins.min(n, m - offset) if offset >= 0 else builtins.min(n + offset, m))
    rows = prims.iota("int32", input.shape, 0)
    cols = prims.iota("int32", input.shape, 1)
    on = prims.eq(prims.add(rows, prims.full(input.shape, offset, "int32")), cols)
    picked = prims.select(on, input, prims.full(input.shape, 0, input.dtype))
    # Each column's (offset >= 0) or row's one element.
    if offset >= 0:
        line = prims.reduce_sum(picked, (0,), input.dtype)
        return prims.slice(line, (offset,), (offset + length,))
    line = prims.reduce_sum(picked, (1,), input.dtype)
    return prims.slice(line, (-offset,), (-offset + length,))


def pad(input, pad, mode="constant", value=0):
    """``input`` padded (``torch.nn.functional.pad``): ``pad`` pairs of
    before and after sizes, the last dimension's first; ``"constant"``
    with ``value``, or ``"circular"`` (wrapping around, as many times as
    it takes)."""
    out = _lift(input)
    for i in range(len(pad) // 2):
        d = out.ndim - 1 - i
        before, after = pad[2 * i], pad[2 * i + 1]
        n = out.shape[d]
        if mode == "constant":
            side = lambda k: prims.full(tuple(k if j == d else s for j, s in enumerate(out.shape)), value, out.dtype)
            parts = ([side(before)] if before > 0 else []) + [out] + ([side(after)] if after > 0 else [])
        elif mode == "circular" and (before > n or after > n):
            # Wider than the dimension (numpy's "wrap"): it repeated, then
            # the window.
            k = -(-before // n)
            repeated = prims.concatenate([out] * (k + 1 + -(-after // n)), d)
            out = repeated.narrow(d, k * n - before, n + before + after)
            continue
        elif mode == "circular":
            part = lambda a, b: out.narrow(d, a, b - a)
            parts = ([part(n - before, n)] if before > 0 else []) + [out] + ([part(0, after)] if after > 0 else [])
        else:
            raise NotImplementedError(f"pad mode {mode!r}")
        out = prims.concatenate(parts, d) if len(parts) > 1 else out
    return out


def gather(input, dim, index):
    """``input``'s elements along ``dim`` at ``index`` (of its rank):
    ``out[i][j] = input[i][index[i][j]]`` for ``dim=1``
    (``torch.gather``, ``numpy.take_along_axis``)."""
    input, index = _lift(input), _lift(index)
    return prims.take_along_axis(input, index, _dim(dim, input.ndim))


def einsum(equation, *operands):
    """A contraction of two operands (``torch.einsum``): no index repeated
    within one, the output's indices the inputs' batch and free ones."""
    inputs, output = equation.replace(" ", "").split("->")
    a, b = inputs.split(",")
    x, y = (_lift(t) for t in operands)
    contracting = [c for c in a if c in b and c not in output]
    batch = [c for c in a if c in b and c in output]
    numbers = (
        ([a.index(c) for c in contracting], [b.index(c) for c in contracting]),
        ([a.index(c) for c in batch], [b.index(c) for c in batch]),
    )
    out = prims.dot_general(x, y, numbers, _accum_dtype(x.dtype), x.dtype)
    order = batch + [c for c in a if c not in b] + [c for c in b if c not in a]
    return out.permute(tuple(order.index(c) for c in output)) if order != list(output) else out
