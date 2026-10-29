"""Python wrappers that record each lumen primitive into the graph being traced."""

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
    "host_wait",
    "device_wait",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    "broadcast",
    "reduce",
    "gather_to",
    "scatter_from",
    "send",
    "recv",
    "to_host",
    "to_device",
    "broadcast_in_dim",
    "transpose",
    "slice",
    "dynamic_slice",
    "dynamic_update_slice",
    "gather",
    "take_along_axis",
    "scatter_add",
    "concatenate",
    "full",
    "iota",
    "random_bits",
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
    "log1p",
    "expm1",
    "exp2",
    "log2",
    "rsqrt",
    "cbrt",
    "erf_inv",
    "is_finite",
    "abs",
    "sign",
    "not_",
    "population_count",
    "clz",
    "pow",
    "rem",
    "min",
    "and_",
    "or_",
    "xor",
    "shift_left",
    "shift_right_logical",
    "shift_right_arithmetic",
    "nextafter",
    "reduce_precision",
]


def bind(name, *operands, **params):
    """Record ``name(*operands, **params)`` in the graph being traced."""
    graph = tracer.current_graph()
    operands = [tracer._lift(x) for x in operands]
    host = tracer._HOSTS[-1] if tracer._HOSTS else None
    # Scalar-only work may run on the host graph; device ops read host values as imported scalar inputs.
    if host is not None:
        out = host.bind(name, operands, params)
        if out is not None:
            same = len(operands) == 1 and (out.dtype, out.shape) == (operands[0].dtype, operands[0].shape)
            out.weak = same and operands[0].weak
            return out
        operands = [host.imported(x) if host.owns(x) else x for x in operands]
    for x in operands:
        if not isinstance(x, tracer.TracedTensor) or x.graph is not graph:
            if hasattr(x, "_not_an_operand"):
                raise TypeError(x._not_an_operand(name))
            raise TypeError(f"{name}: operands must be traced tensors of the current trace, got {x!r}")
    inputs = [x.var for x in operands]
    var = graph.apply(name, inputs, params)
    tracer._record(name, inputs, params, var)
    out = tracer.TracedTensor(graph, var)
    # A unary op keeps a weak (Python-scalar) operand weak only when its type is unchanged.
    out.weak = len(operands) == 1 and operands[0].weak and graph.type_of(var) == graph.type_of(inputs[0])
    # Ops on constants alone keep a recipe so the host can replay them.
    if all(x.recipe is not None for x in operands):
        out.recipe = (name, params, operands)
    # Results computed from any CPU operand, or copied to the host, live on the CPU.
    if name == "to_host" or (name != "to_device" and any(tracer._device_type(x.device) == "cpu" for x in operands)):
        out.device = "cpu:0"
    return out


def add(x, y):
    """Elementwise ``x + y``."""
    return bind("add", x, y)


def sub(x, y):
    """Elementwise ``x - y``."""
    return bind("sub", x, y)


def mul(x, y):
    """Elementwise ``x * y``."""
    return bind("mul", x, y)


def div(x, y):
    """Division; truncating for integers."""
    return bind("div", x, y)


def max(x, y):
    """Elementwise maximum, propagating NaN."""
    return bind("max", x, y)


def eq(x, y):
    """Elementwise ``x == y`` as bool."""
    return bind("eq", x, y)


def lt(x, y):
    """Elementwise ``x < y`` as bool."""
    return bind("lt", x, y)


def neg(x):
    """Elementwise ``-x``."""
    return bind("neg", x)


def exp(x):
    """Elementwise ``e**x``."""
    return bind("exp", x)


def log(x):
    """Elementwise natural logarithm."""
    return bind("log", x)


def sqrt(x):
    """Elementwise square root."""
    return bind("sqrt", x)


def tanh(x):
    """Elementwise hyperbolic tangent."""
    return bind("tanh", x)


def logistic(x):
    """Elementwise sigmoid ``1 / (1 + e**-x)``."""
    return bind("logistic", x)


def cast(x, new_dtype):
    """``x`` converted elementwise to ``new_dtype``."""
    return bind("cast", x, new_dtype=new_dtype)


def select(pred, on_true, on_false):
    """``on_true`` where the bool ``pred`` holds, else ``on_false``."""
    return bind("select", pred, on_true, on_false)


def reduce_sum(x, axes, accum_dtype):
    """The sum over ``axes``, accumulated in ``accum_dtype``, the result's
    dtype: ``x``'s, or ``float32`` for floats narrower than it."""
    return bind("reduce_sum", x, axes=tuple(axes), accum_dtype=accum_dtype)


def reduce_max(x, axes):
    """The maximum over ``axes``, in ``x``'s dtype."""
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


def host_wait(x):
    """``x``, once the device has finished all work queued before it: the
    host blocks until then (PyTorch: ``torch.cuda.synchronize``). In place:
    no copy."""
    return bind("host_wait", x)


def device_wait(x):
    """``x``, once the asynchronous work producing it (a collective) has finished: the device's later work
    waits for it, the host doesn't (PyTorch's ``wait_tensor`` after a functional collective). In place: no
    copy."""
    return bind("device_wait", x)


def all_reduce(x, op, ranks):
    """``x`` combined across the processes ``ranks`` with ``op`` (``"sum"``, ``"max"`` or ``"min"``); every one
    of them gets the result (PyTorch's functional ``all_reduce``; NCCL's ``ncclAllReduce`` on CUDA)."""
    return bind("all_reduce", x, op=op, ranks=tuple(ranks))


def all_gather(x, ranks):
    """Every process in ``ranks``'s ``x`` stacked along a new leading axis, in ``ranks``' order (``ncclAllGather``)."""
    return bind("all_gather", x, ranks=tuple(ranks))


def reduce_scatter(x, op, ranks):
    """``x`` (a leading axis of one row per process in ``ranks``) combined across them with ``op``; the process at
    position ``r`` of ``ranks`` gets row ``r`` (``ncclReduceScatter``)."""
    return bind("reduce_scatter", x, op=op, ranks=tuple(ranks))


def all_to_all(x, ranks):
    """Rows of ``x``'s leading axis (one per process in ``ranks``) exchanged: the process at position ``r`` gets,
    as row ``j``, row ``r`` of the process at position ``j`` (PyTorch's ``all_to_all_single``)."""
    return bind("all_to_all", x, ranks=tuple(ranks))


def broadcast(x, root, ranks):
    """Process ``root``'s ``x``, given to every process in ``ranks`` (``ncclBroadcast``)."""
    return bind("broadcast", x, root=root, ranks=tuple(ranks))


def reduce(x, op, root, ranks):
    """``x`` combined across ``ranks`` with ``op`` on process ``root``; zeros on the others (``ncclReduce``)."""
    return bind("reduce", x, op=op, root=root, ranks=tuple(ranks))


def gather_to(x, root, ranks):
    """Every process in ``ranks``'s ``x`` stacked along a new leading axis, in ``ranks``' order, on process
    ``root``; zeros on the others."""
    return bind("gather_to", x, root=root, ranks=tuple(ranks))


def scatter_from(x, root, ranks):
    """Row ``r`` of process ``root``'s ``x`` (a leading axis of one row per process in ``ranks``) for the process
    at position ``r``; the other processes' ``x`` is not read."""
    return bind("scatter_from", x, root=root, ranks=tuple(ranks))


def send(x, peer, ranks):
    """Sends ``x`` to process ``peer`` of the group ``ranks``; returns an empty token (``ncclSend``)."""
    return bind("send", x, peer=peer, ranks=tuple(ranks))


def recv(shape, dtype, peer, ranks):
    """A ``dtype`` tensor of ``shape`` received from process ``peer`` of the group ``ranks`` (``ncclRecv``)."""
    out = bind("recv", peer=peer, ranks=tuple(ranks), dtype=dtype, shape=tuple(shape))
    # Each receive is its own message: never replayed on the host like an op on constants.
    out.recipe = None
    return out


def to_host(x):
    """``x`` copied to the host (``Tensor.cpu``): the plan's step waits for
    it and copies it there (none on the CPU), its result if returned; the
    steps after it read it on the plan's device."""
    return bind("to_host", x)


def to_device(x):
    """``x``, a host value, copied to the plan's device (``Tensor.to``): in
    stream order, the host not waiting; a device value as it is."""
    return bind("to_device", x)


def reshape(x, new_sizes):
    """``x``'s elements in row-major order with shape ``new_sizes``."""
    return bind("reshape", x, new_sizes=tuple(new_sizes))


def broadcast_in_dim(x, shape, broadcast_dimensions):
    """Operand dimension ``i`` becomes result dimension
    ``broadcast_dimensions[i]`` (size 1, or the result's size)."""
    return bind("broadcast_in_dim", x, shape=tuple(shape), broadcast_dimensions=tuple(broadcast_dimensions))


def transpose(x, permutation):
    """``x`` with result dimension ``i`` taken from operand dimension ``permutation[i]``."""
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


def gather(x, indices, axis):
    """``x``'s entries along ``axis`` that ``indices`` (int32 or int64, any
    shape) pick: ``x``'s shape with ``axis`` replaced by the indices'
    (``jnp.take``; MLX's ``take``, ``weight[ids]``), each index in the axis
    (one out of it the program's error; clamped into it with
    ``lumen.config.compiler.<device>.safe_kernels``)."""
    return bind("gather", x, indices, axis=axis)


def take_along_axis(x, indices, axis):
    """``x``'s entries along ``axis`` at ``indices`` (integers of ``x``'s
    rank, no larger but along ``axis``), the others where they are
    (``jnp.take_along_axis``, ``torch.gather``)."""
    return bind("take_along_axis", x, indices, axis=axis)


def scatter_add(x, indices, updates, axis):
    """``x`` with ``updates`` (shaped as :func:`gather` of it at ``indices``)
    added at the entries along ``axis`` the indices pick, in range as the
    gather's (``lax.scatter_add``): a gather's gradient. Where nothing reads
    ``x`` after it, added in ``x``'s memory (in place)."""
    return bind("scatter_add", x, indices, updates, axis=axis)


def _custom_call(operands, name, kernel, mutated, overlappable=()):
    """Custom op ``name`` (``lumen.custom_op``), function ``kernel`` (its
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
    """A tensor of ``shape`` and ``dtype`` filled with ``fill_value``."""
    return bind("full", shape=tuple(shape), fill_value=fill_value, dtype=dtype)


def random_bits(seed, start, shape, offset):
    """uint32 random bits of ``shape``: Philox4x32-10 keyed by ``seed``,
    element ``i`` at counter ``start`` plus ``offset`` plus ``i`` (``seed``
    and ``start``: uint64 scalars; XLA's ``RngBitGenerator``, PyTorch's
    ``philox_rand``)."""
    return bind("random_bits", seed, start, shape=tuple(shape), offset=offset)


def linear_cross_entropy(h, w, target):
    """Each row of ``h`` ``[B, D]``'s logits ``h · wᵀ`` (``w`` ``[V, D]``,
    accumulated in float32, never stored) reduced to their log-sum-exp and
    the logit of its class (``target`` ``[B]``, int32 or int64): float32
    ``[B, 2]``, ``[lse, logit]`` a row (Cut Cross-Entropy's forward)."""
    return bind("linear_cross_entropy", h, w, target)


def iota(dtype, shape, dimension):
    """The index along ``dimension`` (``lax.broadcasted_iota``)."""
    return bind("iota", dtype=dtype, shape=tuple(shape), dimension=dimension)


def sin(x):
    """Elementwise sine."""
    return bind("sin", x)


def cos(x):
    """Elementwise cosine."""
    return bind("cos", x)


def tan(x):
    """Elementwise tangent."""
    return bind("tan", x)


def asin(x):
    """Elementwise arcsine."""
    return bind("asin", x)


def acos(x):
    """Elementwise arccosine."""
    return bind("acos", x)


def atan(x):
    """Elementwise arctangent."""
    return bind("atan", x)


def sinh(x):
    """Elementwise hyperbolic sine."""
    return bind("sinh", x)


def cosh(x):
    """Elementwise hyperbolic cosine."""
    return bind("cosh", x)


def asinh(x):
    """Elementwise inverse hyperbolic sine."""
    return bind("asinh", x)


def acosh(x):
    """Elementwise inverse hyperbolic cosine."""
    return bind("acosh", x)


def atanh(x):
    """Elementwise inverse hyperbolic tangent."""
    return bind("atanh", x)


def floor(x):
    """Elementwise round toward negative infinity."""
    return bind("floor", x)


def ceil(x):
    """Elementwise round toward positive infinity."""
    return bind("ceil", x)


def round(x):
    """Elementwise round to nearest integer value."""
    return bind("round", x)


def log1p(x):
    """Elementwise ``log(1 + x)``, accurate near zero."""
    return bind("log1p", x)


def expm1(x):
    """Elementwise ``e**x - 1``, accurate near zero."""
    return bind("expm1", x)


def exp2(x):
    """Elementwise ``2**x``."""
    return bind("exp2", x)


def log2(x):
    """Elementwise base-2 logarithm."""
    return bind("log2", x)


def rsqrt(x):
    """Elementwise ``1 / sqrt(x)``."""
    return bind("rsqrt", x)


def cbrt(x):
    """Elementwise cube root."""
    return bind("cbrt", x)


def erf_inv(x):
    """Elementwise inverse error function."""
    return bind("erf_inv", x)


def is_finite(x):
    """Elementwise bool: neither infinite nor NaN."""
    return bind("is_finite", x)


def abs(x):
    """Elementwise absolute value."""
    return bind("abs", x)


def sign(x):
    """Elementwise sign: -1, 0 or 1 (NaN for NaN)."""
    return bind("sign", x)


def not_(x):
    """Elementwise logical not for bool, bitwise not for integers."""
    return bind("not", x)


def population_count(x):
    """Elementwise number of set bits."""
    return bind("population_count", x)


def clz(x):
    """Elementwise number of leading zero bits."""
    return bind("clz", x)


def pow(x, y):
    """Elementwise ``x**y``."""
    return bind("pow", x, y)


def rem(x, y):
    """Elementwise remainder with the sign of ``x`` (C ``fmod``)."""
    return bind("rem", x, y)


def min(x, y):
    """Elementwise minimum, propagating NaN."""
    return bind("min", x, y)


def and_(x, y):
    """Elementwise logical and for bool, bitwise and for integers."""
    return bind("and", x, y)


def or_(x, y):
    """Elementwise logical or for bool, bitwise or for integers."""
    return bind("or", x, y)


def xor(x, y):
    """Elementwise logical xor for bool, bitwise xor for integers."""
    return bind("xor", x, y)


def shift_left(x, y):
    """Elementwise ``x << y``."""
    return bind("shift_left", x, y)


def shift_right_logical(x, y):
    """Elementwise ``x >> y`` filling with zeros."""
    return bind("shift_right_logical", x, y)


def shift_right_arithmetic(x, y):
    """Elementwise ``x >> y`` filling with the sign bit."""
    return bind("shift_right_arithmetic", x, y)


def nextafter(x, y):
    """Elementwise next representable value after ``x`` toward ``y``."""
    return bind("nextafter", x, y)


def reduce_precision(x, exponent_bits, mantissa_bits):
    """``x`` rounded to a float of ``exponent_bits`` and ``mantissa_bits``,
    of its dtype."""
    return bind("reduce_precision", x, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)
