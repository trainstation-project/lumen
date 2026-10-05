"""The primitive ops, as strict functions on traced tensors (JAX:
``jax.lax``). Each records one node in the graph being traced; its shape
and dtype rules are checked in Rust (``lumen/graph/primitive.rs``) and a
violation raises ``ValueError``. Operands of elementwise primitives must
already share a shape and dtype: broadcasting is the torch-like layer's job
(``lumen/graph/tracer.py``), and neither layer converts dtypes implicitly.
"""

from lumen.graph import tracer

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "max",
    "eq",
    "lt",
    "neg",
    "exp",
    "log",
    "sqrt",
    "tanh",
    "logistic",
    "cast",
    "select",
    "reduce_sum",
    "reduce_max",
    "cumsum",
    "dot_general",
    "reshape",
    "wait",
    "broadcast_in_dim",
    "transpose",
    "slice",
    "dynamic_slice",
    "dynamic_update_slice",
    "concatenate",
    "full",
    "iota",
    "random_bits",
]


def bind(name, *operands, **params):
    """Record ``name(*operands, **params)`` in the graph being traced."""
    graph = tracer.current_graph()
    operands = [tracer._lift(x) for x in operands]
    for x in operands:
        if not isinstance(x, tracer.TracedTensor) or x.graph is not graph:
            raise TypeError(f"{name}: operands must be traced tensors of the current trace, got {x!r}")
    inputs = [x.var for x in operands]
    var = graph.apply(name, inputs, params)
    tracer._record(name, inputs, params, var)
    return tracer.TracedTensor(graph, var)


def add(x, y):
    return bind("add", x, y)


def sub(x, y):
    return bind("sub", x, y)


def mul(x, y):
    return bind("mul", x, y)


def div(x, y):
    """Division; truncating for integers."""
    return bind("div", x, y)


def max(x, y):
    """Elementwise maximum, propagating NaN."""
    return bind("max", x, y)


def eq(x, y):
    return bind("eq", x, y)


def lt(x, y):
    return bind("lt", x, y)


def neg(x):
    return bind("neg", x)


def exp(x):
    return bind("exp", x)


def log(x):
    return bind("log", x)


def sqrt(x):
    return bind("sqrt", x)


def tanh(x):
    return bind("tanh", x)


def logistic(x):
    return bind("logistic", x)


def cast(x, new_dtype):
    return bind("cast", x, new_dtype=new_dtype)


def select(pred, on_true, on_false):
    return bind("select", pred, on_true, on_false)


def reduce_sum(x, axes, accum_dtype):
    """The sum over ``axes``, accumulated in ``accum_dtype``, the result's
    dtype: ``x``'s, or ``float32`` for floats narrower than it."""
    return bind("reduce_sum", x, axes=tuple(axes), accum_dtype=accum_dtype)


def reduce_max(x, axes):
    return bind("reduce_max", x, axes=tuple(axes))


def cumsum(x, axis, reverse, accum_dtype):
    """The inclusive cumulative sum along ``axis`` (from its end if
    ``reverse``), accumulated in ``accum_dtype``, the result's dtype: ``x``'s,
    or ``float32`` for floats narrower than it (``lax.cumsum``)."""
    return bind("cumsum", x, axis=int(axis), reverse=bool(reverse), accum_dtype=accum_dtype)


def dot_general(lhs, rhs, dimension_numbers, accum_dtype, output_dtype):
    """``dimension_numbers = ((lhs_contracting, rhs_contracting),
    (lhs_batch, rhs_batch))``; the result's dimensions are the batch
    dimensions, then the free ones of ``lhs``, then those of ``rhs``. It
    accumulates in ``accum_dtype`` (the operands', or ``float32`` for floats
    narrower than it) and returns ``output_dtype`` (the operands' or
    ``accum_dtype``), each element rounded to it once."""
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    return bind(
        "dot_general",
        lhs,
        rhs,
        lhs_contracting=tuple(lhs_contracting),
        rhs_contracting=tuple(rhs_contracting),
        lhs_batch=tuple(lhs_batch),
        rhs_batch=tuple(rhs_batch),
        accum_dtype=accum_dtype,
        output_dtype=output_dtype,
    )


def wait(x):
    """``x``, once the device work producing it has finished: the host
    blocks until then (PyTorch: ``wait_tensor``). In place: no copy."""
    return bind("wait", x)


def reshape(x, new_sizes):
    return bind("reshape", x, new_sizes=tuple(new_sizes))


def broadcast_in_dim(x, shape, broadcast_dimensions):
    """Operand dimension ``i`` becomes result dimension
    ``broadcast_dimensions[i]`` (size 1, or the result's size)."""
    return bind("broadcast_in_dim", x, shape=tuple(shape), broadcast_dimensions=tuple(broadcast_dimensions))


def transpose(x, permutation):
    return bind("transpose", x, permutation=tuple(permutation))


def slice(x, start_indices, limit_indices):
    """The elements from ``start_indices`` up to ``limit_indices``
    (exclusive) in each dimension (``lax.slice``, unit strides)."""
    return bind("slice", x, start_indices=tuple(start_indices), limit_indices=tuple(limit_indices))


def _start_indices(start_indices):
    """Start indices as the primitives take them: a scalar each, Python
    ints as constants of the traced ones' dtype (int32 if none)."""
    traced = [i for i in start_indices if not isinstance(i, int)]
    dtype = tracer._lift(traced[0]).dtype if traced else "int32"
    return [full((), i, dtype) if isinstance(i, int) else i for i in start_indices]


def dynamic_slice(x, start_indices, slice_sizes):
    """The ``slice_sizes`` block of ``x`` at ``start_indices`` (one integer
    scalar, or int, a dimension, read when it runs), each clamped so the
    block is inside ``x`` (``lax.dynamic_slice``)."""
    return bind("dynamic_slice", x, *_start_indices(start_indices), slice_sizes=tuple(slice_sizes))


def dynamic_update_slice(x, update, start_indices):
    """``x`` with the block ``update`` written at ``start_indices`` (as
    :func:`dynamic_slice`'s) (``lax.dynamic_update_slice``); where nothing
    reads ``x`` after it, written in ``x``'s memory alone (in place: a
    donated input's, a weight assigned with ``copy_``)."""
    return bind("dynamic_update_slice", x, update, *_start_indices(start_indices))


def _custom_call(operands, name, kernel, mutated, overlappable=()):
    """Custom op ``name`` (``lumen.ops.custom_op``), function ``kernel`` (its
    handle) on ``operands``, mutating those at ``mutated``, each ``(m, j)``
    of ``overlappable`` the mutated operand ``m`` perhaps in operand ``j``'s
    memory: their new values."""
    first = bind(
        "custom_call",
        *operands,
        op=name,
        kernel=kernel,
        mutated=tuple(mutated),
        overlappable=[tuple(pair) for pair in overlappable],
    )
    rest = [
        bind("fusion_output", first, index=k, dtype=operands[m].dtype, shape=tuple(operands[m].shape))
        for k, m in enumerate(mutated[1:], 1)
    ]
    return [first, *rest]


def concatenate(operands, dimension):
    """``operands`` one after another along ``dimension``, their other
    dimensions equal (``lax.concatenate``)."""
    return bind("concatenate", *operands, dimension=dimension)


def full(shape, fill_value, dtype):
    return bind("full", shape=tuple(shape), fill_value=fill_value, dtype=dtype)


def random_bits(seed, start, shape, offset):
    """uint32 random bits of ``shape``: Philox4x32-10 keyed by ``seed``,
    element ``i`` at counter ``start`` plus ``offset`` plus ``i`` (``seed``
    and ``start``: uint64 scalars; XLA's ``RngBitGenerator``, PyTorch's
    ``philox_rand``)."""
    return bind("random_bits", seed, start, shape=tuple(shape), offset=offset)


def iota(dtype, shape, dimension):
    """The index along ``dimension`` (``lax.broadcasted_iota``)."""
    return bind("iota", dtype=dtype, shape=tuple(shape), dimension=dimension)
