"""Each primitive's JVP rule and, for a linear one, its transpose rule
(JAX: ``jax/_src/lax/lax.py``)."""

from lumen.autograd.core import UndefinedPrimal, _full, defjvp, deflinear, primitive_jvps, primitive_transposes
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
deflinear(prims.cast, lambda ct, x, new_dtype: [prims.cast(ct, x.dtype)])
deflinear(prims.reshape, lambda ct, x, new_sizes: [prims.reshape(ct, x.shape)])
# A wait passes its cotangent through (PyTorch: wait_tensor's backward).
deflinear(prims.wait, lambda ct, x: [ct])


def _transpose_transpose(ct, x, permutation):
    inverse = sorted(range(len(permutation)), key=permutation.__getitem__)
    return [prims.transpose(ct, inverse)]


deflinear(prims.transpose, _transpose_transpose)


def _reduce_sum_transpose(ct, x, axes, accum_dtype):
    ct = prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct
    kept = [d for d in range(x.ndim) if d not in axes]
    return [prims.broadcast_in_dim(ct, x.shape, kept)]


deflinear(prims.reduce_sum, _reduce_sum_transpose)


def _cumsum_transpose(ct, x, axis, reverse, accum_dtype):
    # Each element reaches the outputs after it (before it, reversed): the
    # cotangent summed the other way, in its dtype, then x's.
    ct = prims.cumsum(ct, axis, not reverse, ct.dtype)
    return [prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct]


deflinear(prims.cumsum, _cumsum_transpose)


def _broadcast_in_dim_transpose(ct, x, shape, broadcast_dimensions):
    # Summed over the dimensions it adds and those it stretches from 1.
    dims = broadcast_dimensions
    stretched = [dims[i] for i, n in enumerate(x.shape) if n == 1 and shape[dims[i]] != 1]
    axes = sorted([d for d in range(len(shape)) if d not in dims] + stretched)
    if axes:
        ct = prims.reduce_sum(ct, axes, tracer._accum_dtype(ct.dtype))
        ct = prims.cast(ct, x.dtype) if ct.dtype != x.dtype else ct
    return [prims.reshape(ct, x.shape)]


deflinear(prims.broadcast_in_dim, _broadcast_in_dim_transpose)


def _slice_transpose(ct, x, start_indices, limit_indices):
    # Padded with zeros, one dimension at a time.
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
    pred, x, y = primals
    _, tx, ty = tangents
    if tx is None and ty is None:
        return None
    return prims.select(pred, tx if tx is not None else _full(x, 0), ty if ty is not None else _full(y, 0))


def _select_transpose(ct, pred, x, y):
    zeros = _full(ct, 0)
    return [
        None,
        prims.select(pred, ct, zeros) if isinstance(x, UndefinedPrimal) else None,
        prims.select(pred, zeros, ct) if isinstance(y, UndefinedPrimal) else None,
    ]


primitive_jvps[prims.select] = _select_jvp
primitive_transposes[prims.select] = _select_transpose


# Dynamic slices: linear in the operand (and update), their start indices
# (integers) fixed.
def _dynamic_slice_jvp(primals, tangents, out, slice_sizes):
    x, *indices = primals
    if tangents[0] is None:
        return None
    return prims.dynamic_slice(tangents[0], indices, slice_sizes)


def _dynamic_slice_transpose(ct, x, *indices, slice_sizes):
    # The block's cotangent where it was read, zeros elsewhere.
    return [prims.dynamic_update_slice(_full(x, 0), ct, indices), *[None] * len(indices)]


def _dynamic_update_slice_jvp(primals, tangents, out):
    x, update, *indices = primals
    tx, tu = tangents[:2]
    if tx is None and tu is None:
        return None
    tx, tu = (t if t is not None else _full(p, 0) for t, p in ((tx, x), (tu, update)))
    return prims.dynamic_update_slice(tx, tu, indices)


def _dynamic_update_slice_transpose(ct, x, update, *indices):
    # The operand's cotangent but where the update overwrote it; the
    # update's, the block it was written to.
    return [
        prims.dynamic_update_slice(ct, _full(update, 0), indices) if isinstance(x, UndefinedPrimal) else None,
        prims.dynamic_slice(ct, indices, update.shape) if isinstance(update, UndefinedPrimal) else None,
        *[None] * len(indices),
    ]


primitive_jvps[prims.dynamic_slice] = _dynamic_slice_jvp
primitive_transposes[prims.dynamic_slice] = _dynamic_slice_transpose
primitive_jvps[prims.dynamic_update_slice] = _dynamic_update_slice_jvp
primitive_transposes[prims.dynamic_update_slice] = _dynamic_update_slice_transpose

# Bilinear: linear in each operand, the other fixed.
defjvp(prims.mul, lambda t, out, x, y: prims.mul(t, y), lambda t, out, x, y: prims.mul(x, t))


def _mul_transpose(ct, x, y):
    if isinstance(x, UndefinedPrimal):
        return [prims.mul(ct, y), None]
    return [None, prims.mul(x, ct)]


primitive_transposes[prims.mul] = _mul_transpose
# d(x / y) = dx / y - dy * (out / y): linear in the numerator.
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
    # Shared equally by the elements equal to the max.
    kept = [d for d in range(x.ndim) if d not in axes]
    where = prims.cast(prims.eq(x, prims.broadcast_in_dim(out, x.shape, kept)), x.dtype)
    count = prims.reduce_sum(where, axes, x.dtype)
    return prims.div(prims.reduce_sum(prims.mul(t, where), axes, x.dtype), count)


defjvp(prims.reduce_max, _reduce_max_jvp)


def _dot(x, y, dims, params, output_dtype):
    (lc, rc), (lb, rb) = dims
    return prims.dot_general(x, y, ((lc, rc), (lb, rb)), params["accum_dtype"], output_dtype)


def _dot_dims(params):
    p = params
    return (p["lhs_contracting"], p["rhs_contracting"]), (p["lhs_batch"], p["rhs_batch"])


defjvp(
    prims.dot_general,
    lambda t, out, x, y, **p: _dot(t, y, _dot_dims(p), p, p["output_dtype"]),
    lambda t, out, x, y, **p: _dot(x, t, _dot_dims(p), p, p["output_dtype"]),
)


def _ranges_like(*xs):
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
    # In the operands' dtype (a wider cotangent cast to it).
    ct = prims.cast(ct, y.dtype) if ct.dtype != y.dtype else ct
    x_bar = _dot(ct, y, ((ans_y, y_kept), (ans_batch, y_batch)), params, x.dtype)
    return x_bar if out_axes == list(range(len(out_axes))) else prims.transpose(x_bar, out_axes)


def _dot_general_transpose(ct, x, y, **p):
    (xc, yc), (xb, yb) = _dot_dims(p)
    if isinstance(x, UndefinedPrimal):
        return [_dot_transpose_lhs(ct, x, y, ((xc, yc), (xb, yb)), p), None]
    return [None, _dot_transpose_lhs(ct, y, x, ((yc, xc), (yb, xb)), p, swap_ans=True)]


primitive_transposes[prims.dot_general] = _dot_general_transpose
