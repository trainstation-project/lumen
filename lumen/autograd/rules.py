"""JVP and transpose rules for each differentiable primitive, registered on import."""

from lumen.autograd.core import (
    UndefinedPrimal,
    _full,
    _value,
    defjvp,
    deflinear,
    primitive_jvps,
    primitive_transposes,
)
from lumen.graph import prims, tracer

deflinear(
    prims.add,
    lambda ct, x, y: [ct if isinstance(x, UndefinedPrimal) else None, ct if isinstance(y, UndefinedPrimal) else None],
)
deflinear(
    prims.sub,
    lambda ct, x, y: [
        ct if isinstance(x, UndefinedPrimal) else None,
        prims.neg(ct) if isinstance(y, UndefinedPrimal) else None,
    ],
)
deflinear(prims.neg, lambda ct, x: [prims.neg(ct)])


def _widened(v, dtype):
    """Value ``v``, a gradient's dots rounding their accumulator (``dtype``)
    narrower, maybe transposed or summed (``x @ w.bfloat16()``'s gradient,
    ``w``'s dot's ``x``; ``x``'s of several dots), or their sum in it cast
    narrower, computed with the dots writing their accumulator instead;
    None if it is not one."""
    node = tracer.producer(v)
    if node is None:
        return None
    name, ins, p = node
    if name == "transpose":
        inner = _widened(ins[0], dtype)
        return None if inner is None else prims.transpose(inner, p["permutation"])
    if name == "add":
        x, y = (_widened(i, dtype) for i in ins)
        return None if x is None or y is None else prims.add(x, y)
    if name == "dot_general" and p["accum_dtype"] == dtype and p["output_dtype"] != dtype:
        return _dot(_value(ins[0]), _value(ins[1]), _dot_dims(p), p, dtype)
    if name == "cast" and _value(ins[0]).dtype == dtype:
        return _value(ins[0])
    return None


def _cast_transpose(ct, x, new_dtype):
    """``ct`` cast back to ``x``'s dtype; a gradient's dots rounding their
    accumulator to the narrower dtype only for this (``x @ w.bfloat16()``,
    ``w`` float32: its gradient, its dot's), summed or transposed: those
    dots writing their accumulator, ``x``'s dtype, instead (one rounding
    fewer: Megatron-LM's weight gradients accumulated in float32)."""
    widened = _widened(ct.var, x.dtype) if x.dtype != ct.dtype else None
    return [prims.cast(ct, x.dtype) if widened is None else widened]


deflinear(prims.cast, _cast_transpose)
deflinear(prims.reshape, lambda ct, x, new_sizes: [prims.reshape(ct, x.shape)])
deflinear(prims.wait, lambda ct, x: [ct])
deflinear(prims.to_host, lambda ct, x: [prims.to_device(ct)])
deflinear(prims.to_device, lambda ct, x: [prims.to_host(ct)])


def _transpose_transpose(ct, x, permutation):
    """Transposes ``ct`` back by the inverse permutation."""
    inverse = sorted(range(len(permutation)), key=permutation.__getitem__)
    return [prims.transpose(ct, inverse)]


deflinear(prims.transpose, _transpose_transpose)


def _reduce_sum_transpose(ct, x, axes, accum_dtype):
    """Broadcasts ``ct`` back over the summed ``axes`` in ``x``'s dtype."""
    ct = prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct
    kept = [d for d in range(x.ndim) if d not in axes]
    return [prims.broadcast_in_dim(ct, x.shape, kept)]


deflinear(prims.reduce_sum, _reduce_sum_transpose)


def _summed(name, op):
    """Rejects differentiating a collective that combines with ``op`` other than a sum."""
    if op != "sum":
        raise NotImplementedError(f"{name}: only a sum is differentiable, got op={op!r}")


def _all_reduce_jvp(primals, tangents, out, op, ranks):
    """The tangents summed across ``ranks``, as the values are."""
    _summed("all_reduce", op)
    return None if tangents[0] is None else prims.all_reduce(tangents[0], op, ranks)


def _all_reduce_transpose(ct, x, op, ranks):
    """Every rank's cotangent summed across ``ranks``: each rank's value reached every result."""
    _summed("all_reduce", op)
    return [prims.all_reduce(ct, op, ranks)]


def _all_gather_transpose(ct, x, ranks):
    """Each rank's row of the gathered cotangent, summed across ``ranks``."""
    return [prims.reduce_scatter(ct, "sum", ranks)]


def _reduce_scatter_jvp(primals, tangents, out, op, ranks):
    """The tangents reduce-scattered, as the values are."""
    _summed("reduce_scatter", op)
    return None if tangents[0] is None else prims.reduce_scatter(tangents[0], op, ranks)


def _reduce_scatter_transpose(ct, x, op, ranks):
    """Every rank's cotangent gathered: each row of ``x`` reached the rank that kept it."""
    _summed("reduce_scatter", op)
    return [prims.all_gather(ct, ranks)]


primitive_jvps[prims.all_reduce] = _all_reduce_jvp
primitive_transposes[prims.all_reduce] = _all_reduce_transpose
deflinear(prims.all_gather, _all_gather_transpose)
primitive_jvps[prims.reduce_scatter] = _reduce_scatter_jvp
primitive_transposes[prims.reduce_scatter] = _reduce_scatter_transpose


def _cumsum_transpose(ct, x, axis, reverse, accum_dtype):
    """Cumulative sum of ``ct`` in the opposite direction, cast to ``x``'s dtype."""
    ct = prims.cumsum(ct, axis, not reverse, ct.dtype)
    return [prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct]


deflinear(prims.cumsum, _cumsum_transpose)


def _broadcast_in_dim_transpose(ct, x, shape, broadcast_dimensions):
    """Sums ``ct`` over the added and stretched dimensions, back to ``x``'s shape."""
    dims = broadcast_dimensions
    # Size-1 input dims broadcast to a larger size were stretched and must be summed too.
    stretched = [dims[i] for i, n in enumerate(x.shape) if n == 1 and shape[dims[i]] != 1]
    axes = sorted([d for d in range(len(shape)) if d not in dims] + stretched)
    if axes:
        ct = prims.reduce_sum(ct, axes, tracer._accum_dtype(ct.dtype))
        ct = prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct
    return [prims.reshape(ct, x.shape)]


deflinear(prims.broadcast_in_dim, _broadcast_in_dim_transpose)


def _slice_transpose(ct, x, start_indices, limit_indices):
    """Zero-pads ``ct`` on each dimension back to ``x``'s shape."""
    for d, (start, limit) in enumerate(zip(start_indices, limit_indices)):
        parts = []
        for size in (start, x.shape[d] - limit):
            shape = list(ct.shape)
            shape[d] = size
            parts.append(prims.full(shape, 0.0, ct.dtype) if size else None)
        operands = [p for p in (parts[0], ct, parts[1]) if p is not None]
        ct = prims.concatenate(operands, d) if len(operands) > 1 else ct
    return [ct]


deflinear(prims.slice, _slice_transpose)


def _concatenate_transpose(ct, *operands, dimension):
    """Slices ``ct`` along ``dimension`` into each linear operand's cotangent."""
    cts, offset = [], 0
    for x in operands:
        size = x.shape[dimension]
        if isinstance(x, UndefinedPrimal):
            start = [0] * ct.ndim
            limit = list(ct.shape)
            start[dimension], limit[dimension] = offset, offset + size
            cts.append(prims.slice(ct, start, limit))
        else:
            cts.append(None)
        offset += size
    return cts


deflinear(prims.concatenate, _concatenate_transpose)


def _select_jvp(primals, tangents, out, **params):
    """Selects between the branch tangents (zeros for a branch without one)."""
    pred, x, y = primals
    _, tx, ty = tangents
    if tx is None and ty is None:
        return None
    return prims.select(pred, tx if tx is not None else _full(x, 0), ty if ty is not None else _full(y, 0))


def _select_transpose(ct, pred, x, y):
    """Routes ``ct`` to the branch ``pred`` chose; none for ``pred``."""
    zeros = _full(ct, 0)
    return [
        None,
        prims.select(pred, ct, zeros) if isinstance(x, UndefinedPrimal) else None,
        prims.select(pred, zeros, ct) if isinstance(y, UndefinedPrimal) else None,
    ]


primitive_jvps[prims.select] = _select_jvp
primitive_transposes[prims.select] = _select_transpose


def _dynamic_slice_jvp(primals, tangents, out, slice_sizes):
    """The same dynamic slice of ``x``'s tangent."""
    x, *indices = primals
    if tangents[0] is None:
        return None
    return prims.dynamic_slice(tangents[0], indices, slice_sizes)


def _dynamic_slice_transpose(ct, x, *indices, slice_sizes):
    """Writes ``ct`` into zeros of ``x``'s shape at the slice's indices."""
    return [prims.dynamic_update_slice(_full(x, 0), ct, indices), *[None] * len(indices)]


def _dynamic_update_slice_jvp(primals, tangents, out):
    """Writes the update's tangent into the operand's tangent (zeros for one without)."""
    x, update, *indices = primals
    tx, tu = tangents[:2]
    if tx is None and tu is None:
        return None
    tx, tu = (t if t is not None else _full(p, 0) for t, p in ((tx, x), (tu, update)))
    return prims.dynamic_update_slice(tx, tu, indices)


def _dynamic_update_slice_transpose(ct, x, update, *indices):
    """Splits ``ct`` into the operand's (slice zeroed) and the update's (the slice)."""
    return [
        prims.dynamic_update_slice(ct, _full(update, 0), indices) if isinstance(x, UndefinedPrimal) else None,
        prims.dynamic_slice(ct, indices, update.shape) if isinstance(update, UndefinedPrimal) else None,
        *[None] * len(indices),
    ]


def _gather_jvp(primals, tangents, out, axis):
    """Gathers ``x``'s tangent at the same indices."""
    if tangents[0] is None:
        return None
    return prims.gather(tangents[0], primals[1], axis)


def _gather_transpose(ct, x, indices, axis):
    """Scatter-adds ``ct`` into zeros of ``x``'s shape, accumulating in the wider accumulator dtype."""
    accum = tracer._accum_dtype(x.dtype)
    ct = ct if ct.dtype == accum else prims.cast(ct, accum)
    out = prims.scatter_add(prims.full(x.shape, 0, accum), indices, ct, axis)
    return [out if accum == x.dtype else prims.cast(out, x.dtype), None]


def _scatter_add_jvp(primals, tangents, out, axis):
    """Scatter-adds the updates' tangent into the operand's (zeros for one without)."""
    x, indices, updates = primals
    tx, tu = tangents[0], tangents[2]
    if tx is None and tu is None:
        return None
    tx, tu = (t if t is not None else _full(p, 0) for t, p in ((tx, x), (tu, updates)))
    return prims.scatter_add(tx, indices, tu, axis)


def _scatter_add_transpose(ct, x, indices, updates, axis):
    """Passes ``ct`` to the operand and gathers it at the indices for the updates."""
    return [
        ct if isinstance(x, UndefinedPrimal) else None,
        None,
        prims.gather(ct, indices, axis) if isinstance(updates, UndefinedPrimal) else None,
    ]


def _take_along_axis_jvp(primals, tangents, out, axis):
    """Takes ``x``'s tangent along ``axis`` at the same indices."""
    if tangents[0] is None:
        return None
    return prims.take_along_axis(tangents[0], primals[1], axis)


def _take_along_axis_transpose(ct, x, indices, axis):
    """Scatter-adds ``ct`` into a flattened ``x`` at each index's row-major offset."""
    dtype = "int64" if indices.dtype in ("int64", "uint64") else "int32"
    # Row-major flat offset of every element, with ``indices`` substituted on ``axis``.
    flat, stride = None, 1
    for d in reversed(range(len(x.shape))):
        at = (indices if indices.dtype == dtype else prims.cast(indices, dtype)) if d == axis else None
        at = at if at is not None else prims.iota(dtype, indices.shape, d)
        term = prims.mul(at, prims.full(indices.shape, stride, dtype))
        flat = term if flat is None else prims.add(flat, term)
        stride *= x.shape[d]
    accum = tracer._accum_dtype(x.dtype)
    ct = ct if ct.dtype == accum else prims.cast(ct, accum)
    numel = 1
    for n in x.shape:
        numel *= n
    out = prims.reshape(prims.scatter_add(prims.full((numel,), 0, accum), flat, ct, 0), tuple(x.shape))
    return [out if accum == x.dtype else prims.cast(out, x.dtype), None]


primitive_jvps[prims.gather] = _gather_jvp
primitive_transposes[prims.gather] = _gather_transpose
primitive_jvps[prims.take_along_axis] = _take_along_axis_jvp
primitive_transposes[prims.take_along_axis] = _take_along_axis_transpose
primitive_jvps[prims.scatter_add] = _scatter_add_jvp
primitive_transposes[prims.scatter_add] = _scatter_add_transpose
primitive_jvps[prims.dynamic_slice] = _dynamic_slice_jvp
primitive_transposes[prims.dynamic_slice] = _dynamic_slice_transpose
primitive_jvps[prims.dynamic_update_slice] = _dynamic_update_slice_jvp
primitive_transposes[prims.dynamic_update_slice] = _dynamic_update_slice_transpose

defjvp(prims.mul, lambda t, out, x, y: prims.mul(t, y), lambda t, out, x, y: prims.mul(x, t))


def _mul_transpose(ct, x, y):
    """Multiplies ``ct`` by whichever operand is not the linear one."""
    if isinstance(x, UndefinedPrimal):
        return [prims.mul(ct, y), None]
    return [None, prims.mul(x, ct)]


primitive_transposes[prims.mul] = _mul_transpose
defjvp(
    prims.div, lambda t, out, x, y: prims.div(t, y), lambda t, out, x, y: prims.mul(t, prims.neg(prims.div(out, y)))
)
primitive_transposes[prims.div] = lambda ct, x, y: [prims.div(ct, y), None]


def _max_share(x, y):
    """Where ``x`` is the max: 1; where tied, 0.5 (JAX's balanced share)."""
    one, half, zero = _full(x, 1), _full(x, 0.5), _full(x, 0)
    return prims.select(prims.lt(y, x), one, prims.select(prims.eq(x, y), half, zero))


defjvp(
    prims.max, lambda t, out, x, y: prims.mul(t, _max_share(x, y)), lambda t, out, x, y: prims.mul(t, _max_share(y, x))
)
defjvp(prims.exp, lambda t, out, x: prims.mul(t, out))
defjvp(prims.log, lambda t, out, x: prims.div(t, x))
defjvp(prims.sqrt, lambda t, out, x: prims.div(t, prims.add(out, out)))
defjvp(prims.tanh, lambda t, out, x: prims.mul(t, prims.sub(_full(out, 1), prims.mul(out, out))))
defjvp(prims.logistic, lambda t, out, x: prims.mul(t, prims.mul(out, prims.sub(_full(out, 1), out))))


def _reduce_max_jvp(t, out, x, axes):
    """Averages the tangent over positions equal to the max (ties share evenly)."""
    kept = [d for d in range(x.ndim) if d not in axes]
    where = prims.cast(prims.eq(x, prims.broadcast_in_dim(out, x.shape, kept)), x.dtype)
    count = prims.reduce_sum(where, axes, x.dtype)
    return prims.div(prims.reduce_sum(prims.mul(t, where), axes, x.dtype), count)


defjvp(prims.reduce_max, _reduce_max_jvp)


def _dot(x, y, dims, params, output_dtype):
    """``dot_general`` of ``x`` and ``y`` with ``dims``, ``params``' accumulator and ``output_dtype``."""
    (lc, rc), (lb, rb) = dims
    return prims.dot_general(x, y, ((lc, rc), (lb, rb)), params["accum_dtype"], output_dtype)


def _dot_dims(params):
    """``((lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch))`` from dot params."""
    p = params
    return (p["lhs_contracting"], p["rhs_contracting"]), (p["lhs_batch"], p["rhs_batch"])


defjvp(
    prims.dot_general,
    lambda t, out, x, y, **p: _dot(t, y, _dot_dims(p), p, p["output_dtype"]),
    lambda t, out, x, y, **p: _dot(x, t, _dot_dims(p), p, p["output_dtype"]),
)


def _ranges_like(*xs):
    """Consecutive index ranges, one per sequence in ``xs``, of matching lengths."""
    start = 0
    for x in xs:
        yield list(range(start, start + len(x)))
        start += len(x)


def _dot_transpose_lhs(ct, x, y, dims, params, swap_ans=False):
    """JAX's ``_dot_general_transpose_lhs``: ``x``'s cotangent, a dot of
    ``ct`` and ``y``, transposed to ``x``'s dimensions."""
    (x_contract, y_contract), (x_batch, y_batch) = dims
    x_kept = [d for d in range(x.ndim) if d not in x_contract and d not in x_batch]
    y_kept = [d for d in range(y.ndim) if d not in y_contract and d not in y_batch]
    if swap_ans:
        ans_batch, ans_y, _ = _ranges_like(x_batch, y_kept, x_kept)
    else:
        ans_batch, _, ans_y = _ranges_like(x_batch, x_kept, y_kept)
    order = sorted(range(len(y_contract)), key=lambda i: y_contract[i])
    unsorted = list(x_batch) + x_kept + [x_contract[i] for i in order]
    out_axes = sorted(range(len(unsorted)), key=unsorted.__getitem__)
    ct = prims.cast(ct, y.dtype) if ct.dtype != y.dtype else ct
    x_bar = _dot(ct, y, ((ans_y, y_kept), (ans_batch, y_batch)), params, x.dtype)
    return x_bar if out_axes == list(range(len(out_axes))) else prims.transpose(x_bar, out_axes)


def _dot_general_transpose(ct, x, y, **p):
    """Cotangent of whichever dot operand is the ``UndefinedPrimal``."""
    (xc, yc), (xb, yb) = _dot_dims(p)
    if isinstance(x, UndefinedPrimal):
        return [_dot_transpose_lhs(ct, x, y, ((xc, yc), (xb, yb)), p), None]
    return [None, _dot_transpose_lhs(ct, y, x, ((yc, xc), (yb, xb)), p, swap_ans=True)]


primitive_transposes[prims.dot_general] = _dot_general_transpose
