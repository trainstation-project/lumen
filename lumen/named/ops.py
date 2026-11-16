"""Ops on named tensors (``lumen.named``): each resolves axis names into positions while tracing and
applies the ``lumen.functional`` op, or primitive, on the underlying tensor."""

from __future__ import annotations

import builtins
import math
from typing import Any, Callable, Mapping, Sequence

import lumen.functional as F
from lumen import tensor as _factories
from lumen.graph import prims
from lumen.graph.tracer import TracedTensor, _accum_dtype, _common_dtype, _dot, _lift, _require_float
from lumen.named.tensor import NamedTensor, _check_names, _order

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "logical_and",
    "logical_or",
    "where",
    "clamp",
    "neg",
    "abs",
    "sign",
    "square",
    "reciprocal",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log1p",
    "sqrt",
    "rsqrt",
    "tanh",
    "sigmoid",
    "relu",
    "sin",
    "cos",
    "floor",
    "ceil",
    "round",
    "isfinite",
    "isnan",
    "logical_not",
    "cast",
    "sum",
    "mean",
    "max",
    "min",
    "logsumexp",
    "any",
    "all",
    "argmax",
    "argmin",
    "softmax",
    "log_softmax",
    "dot",
    "contract",
    "rename",
    "align_to",
    "split",
    "merge",
    "index",
    "concatenate",
    "stack",
    "arange",
]


def _named(x: Any, op: str) -> NamedTensor:
    """``x``, or a TypeError naming ``op`` if it is not a named tensor."""
    if not isinstance(x, NamedTensor):
        raise TypeError(f"{op} takes a lumen.named.NamedTensor, got {type(x).__name__}: name its axes first")
    return x


def _axes(x: NamedTensor, axis: Any, op: str) -> tuple[str, ...]:
    """The names ``axis`` (a name, a sequence of names, or None for every
    axis) picks of ``x``; a KeyError for one it does not have."""
    if axis is None:
        return x.names
    names = _check_names((axis,) if isinstance(axis, str) else axis, op)
    unknown = [n for n in names if n not in x.names]
    if unknown:
        raise KeyError(f"{op}: no axis {', '.join(map(repr, unknown))} in a tensor of axes {x.names}")
    return names


def _sizes(operands: Sequence[Any], op: str) -> dict[str, int]:
    """The size of every axis of the named ``operands``, by name, in order of
    first appearance; a ValueError where two disagree. A positional tensor
    among them must be 0-d (a scalar)."""
    sizes = {}
    for x in operands:
        if isinstance(x, NamedTensor):
            for n, s in zip(x.names, x._tensor.shape):
                if sizes.setdefault(n, s) != s:
                    raise ValueError(f"{op}: axis {n!r} has size {sizes[n]} in one operand and {s} in another")
        elif hasattr(x, "shape") and len(x.shape) > 0:
            raise TypeError(
                f"{op}: an operand of shape {list(x.shape)} has no axis names: wrap it in lumen.named.NamedTensor"
            )
    return sizes


def _expand(x: NamedTensor, names: Sequence[str]) -> Any:
    """``x``'s tensor laid out for ``names`` (a superset of its names): its
    axes in their order there, and a size-1 axis for each name it lacks
    after its first, so positional broadcasting (aligned at the trailing
    axes) matches axes by name."""
    mine = [n for n in names if n in x.names]
    t = x.tensor(*mine)
    sizes = x.shape
    shape = [sizes.get(n, 1) for n in names[names.index(mine[0]) :]] if mine else []
    return t if len(shape) == len(mine) else t.reshape(shape)


def _union(operands: Sequence[Any], op: str) -> tuple[str, ...]:
    """The axes of a broadcast of ``operands`` by name (``_sizes`` checks
    them): the order of the named operand with the most axes (the first
    such), so it is read as stored, then each other name in order of first
    appearance."""
    sizes = _sizes(operands, op)
    named = [x for x in operands if isinstance(x, NamedTensor)]
    widest = builtins.max(named, key=lambda x: x.ndim).names if named else ()
    return tuple(dict.fromkeys((*widest, *sizes)))


def _broadcast(fn: Callable[..., Any], op: str, *operands: Any) -> NamedTensor:
    """``fn`` of ``operands`` (named tensors, scalars) broadcast by name: the
    result has the union of their names (``_union``)."""
    names = _union(operands, op)
    tensors = [_expand(x, names) if isinstance(x, NamedTensor) else x for x in operands]
    return NamedTensor(fn(*tensors), names)


def _binary(fn: Callable[..., Any], name: str) -> Callable[..., NamedTensor]:
    """The elementwise op ``fn`` (of two operands) broadcast by name."""

    def op(x, y):
        return _broadcast(fn, name, x, y)

    op.__name__ = op.__qualname__ = name
    op.__doc__ = (
        f"Elementwise ``{name}`` of ``x`` and ``y`` (named tensors or scalars), broadcast by name: the result has "
        "the union of their axes (stored in the order of the operand with more, then the other's new ones), each "
        "axis' sizes must agree (``lumen.functional``'s dtype rules)."
    )
    return op


def _unary(fn: Callable[[Any], Any], name: str) -> Callable[[NamedTensor], NamedTensor]:
    """The elementwise op ``fn`` of one tensor on a named one: the result has its names."""

    def op(x):
        x = _named(x, name)
        return NamedTensor(fn(x._tensor), x.names)

    op.__name__ = op.__qualname__ = name
    op.__doc__ = f"Elementwise ``{name}`` of named tensor ``x``, named as it is (``lumen.functional``'s rules)."
    return op


add = _binary(F.add, "add")
sub = _binary(F.sub, "sub")
mul = _binary(F.mul, "mul")
div = _binary(F.div, "div")
pow = _binary(F.pow, "pow")
maximum = _binary(F.maximum, "maximum")
minimum = _binary(F.minimum, "minimum")
eq = _binary(F.eq, "eq")
ne = _binary(F.ne, "ne")
lt = _binary(F.lt, "lt")
le = _binary(F.le, "le")
gt = _binary(F.gt, "gt")
ge = _binary(F.ge, "ge")
logical_and = _binary(F.logical_and, "logical_and")
logical_or = _binary(F.logical_or, "logical_or")


def where(condition: Any, x: Any, y: Any) -> NamedTensor:
    """``x`` where ``condition`` holds, else ``y``, the three broadcast by name."""
    return _broadcast(F.where, "where", condition, x, y)


def clamp(x: NamedTensor, min: Any = None, max: Any = None) -> NamedTensor:
    """``x`` clamped to ``[min, max]`` (scalars or named tensors, broadcast by name; None: no bound)."""
    return _broadcast(lambda t, lo, hi: F.clamp(t, lo, hi), "clamp", _named(x, "clamp"), min, max)


def _rsqrt(t: Any) -> Any:
    """Elementwise ``1 / sqrt(t)`` of a floating-point tensor, one primitive."""
    return prims.rsqrt(_require_float(_lift(t), "rsqrt"))


neg = _unary(F.neg, "neg")
abs = _unary(F.abs, "abs")
sign = _unary(F.sign, "sign")
square = _unary(F.square, "square")
reciprocal = _unary(F.reciprocal, "reciprocal")
exp = _unary(F.exp, "exp")
exp2 = _unary(F.exp2, "exp2")
expm1 = _unary(F.expm1, "expm1")
log = _unary(F.log, "log")
log2 = _unary(F.log2, "log2")
log1p = _unary(F.log1p, "log1p")
sqrt = _unary(F.sqrt, "sqrt")
rsqrt = _unary(_rsqrt, "rsqrt")
tanh = _unary(F.tanh, "tanh")
sigmoid = _unary(F.sigmoid, "sigmoid")
relu = _unary(F.relu, "relu")
sin = _unary(F.sin, "sin")
cos = _unary(F.cos, "cos")
floor = _unary(F.floor, "floor")
ceil = _unary(F.ceil, "ceil")
round = _unary(F.round, "round")
isfinite = _unary(F.isfinite, "isfinite")
isnan = _unary(F.isnan, "isnan")
logical_not = _unary(F.logical_not, "logical_not")


def cast(x: NamedTensor, dtype: str) -> NamedTensor:
    """``x`` converted to ``dtype``, named as it is."""
    x = _named(x, "cast")
    return NamedTensor(x._tensor.to(dtype=dtype), x.names)


def _reduction(fn: Callable[[Any, tuple[int, ...]], Any], name: str) -> Callable[..., NamedTensor]:
    """The reduction ``fn(tensor, dims)`` over named axes: the result drops them."""

    def op(x, axis=None):
        x = _named(x, name)
        axes = _axes(x, axis, name)
        dims = tuple(x.names.index(n) for n in axes)
        return NamedTensor(fn(x._tensor, dims), tuple(n for n in x.names if n not in axes))

    op.__name__ = op.__qualname__ = name
    op.__doc__ = (
        f"``{name}`` of named tensor ``x`` over ``axis`` (a name or names; every axis if None), which the "
        "result drops (``lumen.functional``'s dtype rules)."
    )
    return op


def _logsumexp(t: Any, dims: tuple[int, ...]) -> Any:
    """``log(sum(exp(t)))`` over ``dims``, computed from the max-shifted tensor for stability."""
    m = F.amax(t, dims, keepdim=True)
    return F.log(F.sum(F.exp(t - m), dims)) + m.reshape([n for d, n in enumerate(t.shape) if d not in dims])


sum = _reduction(lambda t, dims: F.sum(t, dims), "sum")
mean = _reduction(lambda t, dims: F.mean(t, dims), "mean")
max = _reduction(lambda t, dims: F.amax(t, dims), "max")
min = _reduction(lambda t, dims: F.amin(t, dims), "min")
logsumexp = _reduction(_logsumexp, "logsumexp")
any = _reduction(lambda t, dims: F.any(t, dims), "any")
all = _reduction(lambda t, dims: F.all(t, dims), "all")


def _one(x: NamedTensor, axis: str, op: str) -> int:
    """The position of the one axis ``axis`` (a name) of ``x``."""
    if not isinstance(axis, str):
        raise TypeError(f"{op} works along one axis, a name: got {axis!r}")
    return x.names.index(_axes(x, axis, op)[0])


def argmax(x: NamedTensor, axis: str) -> NamedTensor:
    """The index of the maximum along ``axis`` (a name), which the result drops."""
    d = _one(_named(x, "argmax"), axis, "argmax")
    return NamedTensor(F.argmax(x._tensor, d), x.names[:d] + x.names[d + 1 :])


def argmin(x: NamedTensor, axis: str) -> NamedTensor:
    """The index of the minimum along ``axis`` (a name), which the result drops."""
    d = _one(_named(x, "argmin"), axis, "argmin")
    return NamedTensor(F.argmin(x._tensor, d), x.names[:d] + x.names[d + 1 :])


def softmax(x: NamedTensor, axis: str) -> NamedTensor:
    """``F.softmax`` along ``axis`` (a name), named as ``x`` is."""
    d = _one(_named(x, "softmax"), axis, "softmax")
    return NamedTensor(F.softmax(x._tensor, d), x.names)


def log_softmax(x: NamedTensor, axis: str) -> NamedTensor:
    """``F.log_softmax`` along ``axis`` (a name), named as ``x`` is."""
    d = _one(_named(x, "log_softmax"), axis, "log_softmax")
    return NamedTensor(F.log_softmax(x._tensor, d), x.names)


def _pair(
    a: NamedTensor, b: NamedTensor, over: Sequence[str], batch: Sequence[str], accum: str, output: str
) -> NamedTensor:
    """One ``dot_general`` of ``a`` and ``b``: summing the axes ``over``,
    batched over ``batch`` (both in each), accumulated in ``accum`` and
    rounded to ``output``. Its axes are dot_general's order, so no
    transpose follows: ``batch``, then ``a``'s other axes, then ``b``'s."""
    numbers = (
        ([a.names.index(n) for n in over], [b.names.index(n) for n in over]),
        ([a.names.index(n) for n in batch], [b.names.index(n) for n in batch]),
    )
    shared = set(over) | set(batch)
    names = (*batch, *(n for n in a.names if n not in shared), *(n for n in b.names if n not in shared))
    return NamedTensor(_dot(a._tensor, b._tensor, numbers, accum, output), names)


def _dtypes(op: str, operands: Sequence[NamedTensor], accum_dtype: str | None, output_dtype: str | None):
    """The operands' common dtype, and the accumulation and output dtypes:
    by default the matmul accumulation dtype of it (float32 for floats
    narrower) and it."""
    dtype = _common_dtype(op, [x._tensor for x in operands])
    accum = accum_dtype or _accum_dtype(dtype)
    return dtype, accum, output_dtype or dtype


def dot(
    a: NamedTensor,
    b: NamedTensor,
    over: Any,
    *,
    accum_dtype: str | None = None,
    output_dtype: str | None = None,
) -> NamedTensor:
    """``a`` times ``b`` summed over the axes ``over`` (a name or names, in
    both); the other axes both have are batch axes, matched by name, and
    each one's own axes are kept: a matmul, a batched matmul or a linear
    layer by name (``dot(x, w, over="embed")``). One ``dot_general``,
    accumulated in ``accum_dtype`` (default: float32 for float operands,
    float64 and integers their own) and rounded to ``output_dtype``
    (default: the operands' dtype). The result's axis order is
    dot_general's: batch axes, then ``a``'s, then ``b``'s."""
    a, b = _named(a, "dot"), _named(b, "dot")
    over = _check_names((over,) if isinstance(over, str) else over, "dot")
    missing = [n for n in over if n not in a.names or n not in b.names]
    if missing:
        raise KeyError(f"dot: summed axes must be in both operands: {missing} not in both of {a.names} and {b.names}")
    _sizes((a, b), "dot")
    _, accum, output = _dtypes("dot", (a, b), accum_dtype, output_dtype)
    batch = [n for n in a.names if n in b.names and n not in over]
    return _pair(a, b, over, batch, accum, output)


def contract(
    *operands: NamedTensor,
    keep: Any = (),
    accum_dtype: str | None = None,
    output_dtype: str | None = None,
) -> NamedTensor:
    """The einsum of ``operands`` by name: every axis two or more of them
    share is summed, except those in ``keep`` (a name or names), batch axes
    the result keeps with every axis one operand alone has. Two operands
    are one ``dot_general`` (as :func:`dot`); more, a chain of them left to
    right, where an axis a later operand also has stays a batch axis until
    its last operand, and each step but the last is rounded to the
    operands' dtype. Dtypes as :func:`dot`."""
    if len(operands) < 2:
        raise ValueError(f"contract takes two or more named tensors, got {len(operands)}")
    operands = [_named(x, "contract") for x in operands]
    keep = _check_names((keep,) if isinstance(keep, str) else keep, "contract")
    _sizes(operands, "contract")
    dtype, accum, output = _dtypes("contract", operands, accum_dtype, output_dtype)
    out = operands[0]
    for i, nxt in enumerate(operands[1:], 1):
        later = set(keep).union(*(x.names for x in operands[i + 1 :]))
        shared = [n for n in out.names if n in nxt.names]
        last = i == len(operands) - 1
        over = [n for n in shared if n not in later]
        batch = [n for n in shared if n in later]
        out = _pair(out, nxt, over, batch, accum, output if last else dtype)
    return out


def rename(x: NamedTensor, mapping: Mapping[str, str] | None = None, **names: str) -> NamedTensor:
    """``x`` with axes renamed: ``mapping`` and ``names`` take old names to
    new ones (the others keep theirs); names must stay unique."""
    x = _named(x, "rename")
    renames = {**(mapping or {}), **names}
    _axes(x, tuple(renames), "rename")
    return NamedTensor(x._tensor, tuple(renames.get(n, n) for n in x.names))


def align_to(x: NamedTensor, *order: Any) -> NamedTensor:
    """``x`` with its axes stored in ``order`` (``NamedTensor.align_to``)."""
    return _named(x, "align_to").align_to(*order)


def align(x: NamedTensor, like: NamedTensor, op: str) -> Any:
    """``x``'s tensor with its axes in ``like``'s order: both must have the same axes, of the same sizes."""
    x = _named(x, op)
    if set(x.names) != set(like.names):
        raise ValueError(f"{op}: axes {x.names} do not match {like.names}")
    _sizes((x, like), op)
    return x.tensor(*like.names)


def split(x: NamedTensor, axis: str, /, **sizes: int) -> NamedTensor:
    """``x`` with axis ``axis`` split into axes of ``sizes`` (in order, the
    first the outermost, one may be -1 to infer it), where ``axis`` was:
    ``split(x, "embed", heads=8, head_dim=64)``. A reshape."""
    x = _named(x, "split")
    d = _one(x, axis, "split")
    if not sizes:
        raise ValueError("split: name the axes to split into, with their sizes")
    n = x._tensor.shape[d]
    parts = list(sizes.values())
    if parts.count(-1) > 1:
        raise ValueError(f"split: only one size can be inferred, got {sizes}")
    known = math.prod(s for s in parts if s != -1)
    if -1 in parts:
        if known == 0 or n % known:
            raise ValueError(f"split: axis {axis!r} of size {n} does not split into {sizes}")
        parts[parts.index(-1)] = n // known
    if math.prod(parts) != n:
        raise ValueError(f"split: axis {axis!r} of size {n} does not split into {sizes}")
    shape = list(x._tensor.shape)
    names = x.names[:d] + tuple(sizes) + x.names[d + 1 :]
    return NamedTensor(x._tensor.reshape(shape[:d] + parts + shape[d + 1 :]), names)


def merge(x: NamedTensor, axes: Sequence[str], into: str) -> NamedTensor:
    """``x`` with ``axes`` (names, the first the outermost) flattened into
    one axis ``into``. A reshape, after a transpose only if ``axes`` are not
    already adjacent and in that order."""
    x = _named(x, "merge")
    axes = _axes(x, (axes,) if isinstance(axes, str) else axes, "merge")
    if not axes:
        raise ValueError("merge: name the axes to merge")
    # The merged group goes where its first-stored axis is, so an adjacent group needs no transpose.
    at = builtins.min(x.names.index(n) for n in axes)
    before = [n for n in x.names[:at] if n not in axes]
    after = [n for n in x.names[at:] if n not in axes]
    sizes = x.shape
    shape = [sizes[n] for n in before] + [math.prod(sizes[n] for n in axes)] + [sizes[n] for n in after]
    return NamedTensor(x.tensor(*before, *axes, *after).reshape(shape), (*before, into, *after))


def index(x: NamedTensor, selection: Mapping[str, Any]) -> NamedTensor:
    """``x`` indexed by name: ``selection`` maps axis names to an int (the
    axis is dropped), a slice of unit step (it is kept, narrowed), or a
    named tensor of integer indices (a gather: the axis is replaced by the
    indices' axes, ``weight[{"vocab": ids}]``)."""
    x = _named(x, "index")
    if not isinstance(selection, Mapping):
        raise TypeError(f"a named tensor is indexed by name, with a dict: got {selection!r}")
    _axes(x, tuple(selection), "index")
    key, names, gathers = [], [], []
    for n in x.names:
        k = selection.get(n, slice(None))
        if isinstance(k, NamedTensor):
            gathers.append((n, k))
            k = slice(None)
        elif not isinstance(k, (int, slice)) or isinstance(k, bool):
            raise TypeError(f"index: axis {n!r} takes an int, a slice or a named tensor of indices, got {k!r}")
        key.append(k)
        if not isinstance(k, int):
            names.append(n)
    out = NamedTensor(x._tensor[tuple(key)], tuple(names))
    for n, ids in gathers:
        d = out.names.index(n)
        clash = [m for m in ids.names if m in out.names and m != n]
        if clash:
            raise ValueError(f"index: the indices' axes {clash} are already axes of the tensor indexed")
        out = NamedTensor(prims.gather(out._tensor, ids._tensor, d), out.names[:d] + ids.names + out.names[d + 1 :])
    return out


def concatenate(tensors: Sequence[NamedTensor], axis: str) -> NamedTensor:
    """``tensors`` one after another along ``axis``, which each has; their
    other axes must match by name and size (any order: each is aligned to
    the first's)."""
    tensors = [_named(t, "concatenate") for t in tensors]
    if not tensors:
        raise ValueError("concatenate needs at least one tensor")
    first = tensors[0]
    d = _one(first, axis, "concatenate")
    parts = []
    for t in tensors:
        _one(t, axis, "concatenate")
        if set(t.names) != set(first.names):
            raise ValueError(f"concatenate: axes {t.names} do not match {first.names}")
        for n in first.names:
            if n != axis and t.size(n) != first.size(n):
                raise ValueError(
                    f"concatenate: axis {n!r} has size {first.size(n)} in one tensor and {t.size(n)} in another"
                )
        parts.append(t.tensor(*first.names))
    return NamedTensor(F.cat(parts, d), first.names)


def stack(tensors: Sequence[NamedTensor], axis: str) -> NamedTensor:
    """``tensors`` (of the same axes and sizes, in any order) stacked along a new axis ``axis``."""
    tensors = [_named(t, "stack") for t in tensors]
    if not tensors:
        raise ValueError("stack needs at least one tensor")
    first = tensors[0]
    if axis in first.names:
        raise ValueError(f"stack: the new axis {axis!r} is already an axis of {first.names}")
    parts = [align(t, first, "stack") for t in tensors]
    return NamedTensor(F.stack(parts, 0), (axis, *first.names))


def arange(axis: str, n: int, dtype: str | None = None) -> NamedTensor:
    """``0, 1, ..., n - 1`` along an axis named ``axis`` (``lumen.arange``, of ``dtype``)."""
    return NamedTensor(_factories.arange(n, dtype), (axis,))
