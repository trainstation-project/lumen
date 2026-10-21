from lumen.graph import prims, tracer
from lumen.graph.tracer import TracedTensor
from lumen.pallas._C import Kernel

from .call import ANY, ShapeDtypeStruct, pallas_call, program
from .pipeline import (
    Blocked,
    BlockSpec,
    BoundedSlice,
    Element,
    axis_index,
    axis_size,
    emit_pipeline,
    num_programs,
    program_id,
)

__all__ = [
    "ANY",
    "BlockSpec",
    "Blocked",
    "BoundedSlice",
    "Element",
    "Ref",
    "SMEM",
    "SemaphoreType",
    "ShapeDtypeStruct",
    "VMEM",
    "async_copy",
    "axis_index",
    "axis_size",
    "bitcast",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "cdiv",
    "clz",
    "cond",
    "broadcasted_iota",
    "delay",
    "dot_general",
    "dslice",
    "erf_inv",
    "integer_pow",
    "reduce_precision",
    "scan",
    "select_n",
    "shift_right_arithmetic",
    "dot",
    "ds",
    "emit_pipeline",
    "fori_loop",
    "kernel",
    "loop",
    "make_async_copy",
    "multiple_of",
    "num_programs",
    "pallas_call",
    "population_count",
    "program_id",
    "shift_left",
    "shift_right_logical",
    "sparse_core_kernel",
    "switch",
    "when",
    "while_loop",
]


def _kernel():
    """The kernel being traced."""
    k = tracer.current_graph()
    if not isinstance(k, Kernel):
        raise RuntimeError("lumen.pallas ops run inside a kernel (lumen.pallas.kernel)")
    return k


def _i32(x):
    """``x`` (an int, or a traced int32 scalar) as a traced int32 scalar."""
    if isinstance(x, TracedTensor):
        return x
    return prims.full((), int(x), "int32")


def _like(x, v):
    """``v`` (a Python scalar, or a value) as a value of ``x``'s type."""
    if isinstance(v, TracedTensor):
        return v
    return prims.full(x.shape, v, x.dtype)


class ds:
    """A window of a ref's dimension: ``size`` elements from ``start`` (an
    int, or a traced int32 scalar), every ``stride``-th (``pl.ds``,
    ``pl.dslice``)."""

    def __init__(self, start, size=None, stride=None):
        if size is None:
            start, size = 0, start
        self.start, self.size, self.stride = start, size, 1 if stride is None else stride


dslice = ds


def cdiv(a, b):
    """``a / b`` rounded up (ints, or traced int32 scalars)."""
    return (a + b - 1) // b


class VMEM:
    """A VMEM scratch buffer of ``shape`` and ``dtype`` (``pltpu.VMEM``);
    as a ``BlockSpec``'s ``memory_space``, an operand staged in VMEM."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


class SMEM:
    """An SMEM (scalar memory) scratch buffer of ``shape`` and ``dtype``
    (``pltpu.SMEM``); as a ``BlockSpec``'s ``memory_space``, an operand
    staged in SMEM, its elements read and written one at a time."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), dtype


class _Semaphores:
    def __init__(self, shape=()):
        self.shape = tuple(shape)

    def __call__(self, shape):
        return _Semaphores(shape)


class SemaphoreType:
    """``SemaphoreType.DMA``: one DMA semaphore; ``SemaphoreType.DMA(shape)``:
    an array of them (``pltpu.SemaphoreType``)."""

    DMA = _Semaphores()


class _At:
    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, index):
        if isinstance(index, Ref):
            return _Indexed(self.ref, index)
        return self.ref._window(index)


class _Indexed:
    def __init__(self, ref, offsets):
        self.ref, self.offsets, self.kernel = ref, offsets, ref.kernel


class Ref:
    """A buffer of a kernel: its ``dtype``, ``shape`` and ``space`` (``"any"``
    for HBM, ``"vmem"``, ``"smem"``, ``"semaphore"``). ``ref[index]`` loads
    (a VMEM window whole, an SMEM element), ``ref[index] = value`` stores,
    ``ref.at[index]`` is a window of it (a ref), as in Pallas. A ``shape`` of
    ``()`` reads a ref of one element as a scalar (a TPU copies whole rows:
    a scalar's are a ``(1,)``'s)."""

    def __init__(self, kernel, var, shape=None):
        self.kernel, self.var = kernel, var
        self.dtype, physical, self.space = kernel.ref_type(var)
        self.shape = tuple(physical) if shape is None else tuple(shape)
        self._scalar = self.shape == () and tuple(physical) == (1,)

    def __repr__(self):
        return f"Ref(%{self.var}, {self.dtype}{list(self.shape)}, {self.space})"

    def _not_an_operand(self, primitive):
        """Why it is not an operand of ``primitive`` (JAX's message)."""
        shape = ",".join(map(str, self.shape))
        return (
            f"Attempting to pass a Ref <{self.dtype}[{shape}]> to a primitive: {primitive} "
            "-- did you forget to unpack ([...]) the ref?"
        )

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def at(self):
        return _At(self)

    def _window(self, index):
        """The window of ``index``: ints and traced scalars pick an element
        (the dimension squeezed), slices and ``ds`` a range, ``...`` the
        rest. A strided range's window covers it; its strides are
        :meth:`_strided`'s."""
        window, strided = self._strided(index)
        if strided is not None:
            raise IndexError("a ref's window is contiguous: read strided elements with ref[index]")
        return window

    def _strided(self, index, squeezing=True):
        """``index``'s window (whole in its strided dimensions) and, if any
        dimension is strided, each of the window's dimensions' start, count
        and stride. Without ``squeezing``, an element index keeps its
        dimension (of one)."""
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            at = index.index(Ellipsis)
            index = index[:at] + (slice(None),) * (self.ndim - len(index) + 1) + index[at + 1 :]
        index = index + (slice(None),) * (self.ndim - len(index))
        if len(index) != self.ndim:
            raise IndexError(f"{len(index)} indices for a ref of {self.ndim} dimensions")
        starts, dynamic, sizes, squeezed, strides = [], [], [], [], []
        for i, n in zip(index, self.shape):
            if isinstance(i, slice):
                start, stop, step = i.indices(n)
                if step < 1:
                    raise IndexError("ref windows go forwards")
                start, size, squeeze, stride = start, len(range(start, stop, step)), False, step
            elif isinstance(i, ds):
                start, size, squeeze, stride = i.start, i.size, False, i.stride
            else:
                start, size, squeeze, stride = i, 1, True, 1
            if stride != 1:
                strides.append((start, size, stride))
                start, size = 0, n
            elif not squeeze:
                strides.append((0, size, 1))
            if isinstance(start, TracedTensor):
                starts.append(None)
                dynamic.append(start.var)
            else:
                starts.append(int(start))
            sizes.append(size)
            squeezed.append(squeeze and squeezing)
        strided = strides if any(s != 1 for _, _, s in strides) else None
        if starts == [0] * self.ndim and sizes == list(self.shape) and not any(squeezed):
            return self, strided
        return Ref(self.kernel, self.kernel.slice(self.var, starts, dynamic, sizes, squeezed)), strided

    def _element(self, index):
        """An SMEM ref's element ``index``: an index a dimension, or a whole
        dimension of one (``...`` all the rest). Its indices' values and the
        shape it is read as (its whole dimensions' ones)."""
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            at = index.index(Ellipsis)
            index = index[:at] + (slice(None),) * (self.ndim - len(index) + 1) + index[at + 1 :]
        whole = [isinstance(i, slice) and i.indices(n) == (0, 1, 1) for i, n in zip(index, self.shape)]
        if len(index) != self.ndim or any(isinstance(i, (slice, ds)) and not w for i, w in zip(index, whole)):
            raise IndexError(f"an SMEM ref's element takes {self.ndim} indices, got {index!r}")
        indices = [_i32(0 if w else i).var for i, w in zip(index, whole)]
        return (indices or [_i32(0).var]) if self._scalar else indices, (1,) * sum(whole)

    def _access(self, index):
        """``index`` as each dimension's start (an int or a traced int32
        scalar), size, stride, and whether it is squeezed (an element)."""
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            at = index.index(Ellipsis)
            index = index[:at] + (slice(None),) * (self.ndim - len(index) + 1) + index[at + 1 :]
        index = index + (slice(None),) * (self.ndim - len(index))
        if len(index) != self.ndim:
            raise IndexError(f"{len(index)} indices for a ref of {self.ndim} dimensions")
        access = []
        for i, n in zip(index, self.shape):
            if isinstance(i, slice):
                start, stop, step = i.indices(n)
                if step < 1:
                    raise IndexError("ref windows go forwards")
                access.append((start, len(range(start, stop, step)), step, False))
            elif isinstance(i, ds):
                access.append((i.start, i.size, i.stride, False))
            else:
                access.append((i, 1, 1, True))
        return access

    def __getitem__(self, index):
        if self.space == "vmem" and self.dtype != "bool":
            access = self._access(index)
            starts = [_i32(a).var for a, _, _, _ in access]
            sizes = [k for _, k, _, _ in access]
            strides = [s for _, _, s, _ in access]
            value = TracedTensor(self.kernel, self.kernel.load_strided(self.var, starts, sizes, strides))
            shape = [k for _, k, _, squeezed in access if not squeezed]
            return value.reshape(shape) if len(shape) != len(sizes) else value
        if self.space == "smem":
            indices, shape = self._element(index)
            value = TracedTensor(self.kernel, self.kernel.load(self.var, indices))
            return value.reshape(shape) if shape else value
        window, strided = self._strided(index)
        if strided is not None:
            starts = [_i32(a).var for a, _, _ in strided]
            sizes, strides = [k for _, k, _ in strided], [s for _, _, s in strided]
            return TracedTensor(self.kernel, self.kernel.load_strided(window.var, starts, sizes, strides))
        if self.ndim and not window.ndim:
            window, _ = self._strided(index, squeezing=False)
            return TracedTensor(self.kernel, self.kernel.load(window.var)).reshape(())
        return TracedTensor(self.kernel, self.kernel.load(window.var))

    def __setitem__(self, index, value):
        if isinstance(value, Ref):
            raise ValueError("Cannot store a Ref into another Ref. Did you forget to load from it using `[...]`?")
        if hasattr(value, "shape") and not isinstance(value, TracedTensor):
            raise ValueError(
                f"The kernel function of a pallas_call captures constants ({value.dtype}{list(value.shape)}): "
                "pass them to it as operands"
            )
        if self.space == "smem":
            indices, _ = self._element(index)
            value = prims.full((), value, self.dtype) if not isinstance(value, TracedTensor) else value
            self.kernel.store(self.var, value.reshape(()).var, indices)
            return
        if self.space == "vmem" and self.dtype != "bool":
            access = self._access(index)
            sizes = [k for _, k, _, _ in access]
            shape = [k for _, k, _, squeezed in access if not squeezed]
            value = prims.full(shape, value, self.dtype) if not isinstance(value, TracedTensor) else value
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"a store of {list(shape)} elements got a value of {list(value.shape)}")
            value = value.reshape(sizes) if len(shape) != len(sizes) else value
            starts = [_i32(a).var for a, _, _, _ in access]
            self.kernel.store_strided(self.var, value.var, starts, [s for _, _, s, _ in access])
            return
        window, strided = self._strided(index)
        if strided is not None:
            shape = [k for _, k, _ in strided]
            value = prims.full(shape, value, window.dtype) if not isinstance(value, TracedTensor) else value
            starts = [_i32(a).var for a, _, _ in strided]
            self.kernel.store_strided(window.var, value.var, starts, [s for _, _, s in strided])
            return
        value = prims.full(window.shape, value, window.dtype) if not isinstance(value, TracedTensor) else value
        self.kernel.store(window.var, value.var)

    def reshape(self, *shape):
        """Its elements, row-major, as ``shape``."""
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return Ref(self.kernel, self.kernel.reshape(self.var, list(shape)))

    def bitcast(self, dtype):
        """Its bits as ``dtype``, of its width."""
        return Ref(self.kernel, self.kernel.ref_bitcast(self.var, dtype))


def kernel(body, *, scalars, rows, host_rows=None, scratch=(), name="kernel"):
    """``body`` traced as a kernel on a device's memory, lowered to Mosaic:
    ``body(scalars, [host,] memory, result, *scratch)``. ``scalars`` is an
    SMEM ref of that many int32s; ``host`` (if ``host_rows``) host bytes
    copied to the TPU and ``memory`` the device's ``rows`` rows of memory,
    both ``uint32[n, 1, 128]`` in HBM; ``result`` the memory as the kernel
    writes it; then a ref of each ``scratch`` (``VMEM``, ``SemaphoreType``).
    """
    k = Kernel(scalars, rows, host_rows)
    with tracer.tracing(k):
        refs = [Ref(k, v) for v in k.params()]
        refs += [_alloc(k, s) for s in scratch]
        body(*refs)
    return k.lower(name)


def sparse_core_kernel(body, *, refs, mesh, scratch=(), name="kernel"):
    k = Kernel.sparse_core([(dtype, list(shape)) for dtype, shape in refs], list(mesh))
    with tracer.tracing(k):
        params = k.params()
        args = [TracedTensor(k, v) for v in params[:2]] + [Ref(k, v) for v in params[2:]]
        args += [_alloc(k, s) for s in scratch]
        body(*args)
    return k.lower(name)


def _alloc(k, s):
    """A ref of scratch ``s`` (``VMEM``, ``SMEM``, ``SemaphoreType``) of kernel
    ``k``."""
    if isinstance(s, VMEM):
        return Ref(k, k.vmem(s.dtype, list(s.shape)))
    if isinstance(s, SMEM):
        return Ref(k, k.smem(s.dtype, list(s.shape)))
    return Ref(k, k.semaphores(list(s.shape)))


def scratch(s):
    """A ref of scratch ``s`` (``VMEM``, ``SMEM``, ``SemaphoreType``) of the
    kernel being traced, made as it is traced (``emit_pipeline``'s
    buffers)."""
    return _alloc(_kernel(), s)


def when(cond):
    """Runs the decorated function, traced, only where ``cond`` (a bool, or
    a traced bool scalar) holds (``pl.when``)."""

    def decorate(f):
        if not isinstance(cond, TracedTensor):
            if cond:
                f()
            return
        k = _kernel()
        k.begin_if(cond.var)
        f()
        k.end_if()

    return decorate


def _flatten(tree):
    """``tree``'s values (traced tensors, ints as int32 scalars, in lists and
    tuples) and the function rebuilding it from such values."""
    if isinstance(tree, (list, tuple)):
        parts = [_flatten(t) for t in tree]
        leaves = [leaf for ls, _ in parts for leaf in ls]

        def rebuild(values):
            out, at = [], 0
            for ls, r in parts:
                out.append(r(values[at : at + len(ls)]))
                at += len(ls)
            return type(tree)(out)

        return leaves, rebuild
    if tree is None:
        return [], lambda values: None
    return [_i32(tree) if isinstance(tree, int) else tree], lambda values: values[0]


def fori_loop(lower, upper, body, init_val):
    """``body(i, carry)`` for ``i`` in ``lower..upper``, carrying values from
    ``init_val`` (``lax.fori_loop``): a loop of the kernel, traced once."""
    k = _kernel()
    leaves, rebuild = _flatten(init_val)
    i, args = k.begin_for(_i32(lower).var, _i32(upper).var, [v.var for v in leaves])
    carried = rebuild([TracedTensor(k, a) for a in args])
    out, _ = _flatten(body(TracedTensor(k, i), carried))
    results = k.end_for([v.var for v in out])
    return rebuild([TracedTensor(k, r) for r in results])


def cond(pred, true_fun, false_fun, *operands):
    """``true_fun(*operands)`` if ``pred`` (a bool, or a traced bool scalar),
    else ``false_fun(*operands)``, their values of one structure and types
    (``lax.cond``)."""
    if not isinstance(pred, TracedTensor):
        return (true_fun if pred else false_fun)(*operands)
    k = _kernel()
    k.begin_if(pred.var)
    then, rebuild = _flatten(true_fun(*operands))
    k.begin_else([v.var for v in then])
    otherwise, _ = _flatten(false_fun(*operands))
    results = k.end_if([v.var for v in otherwise])
    return rebuild([TracedTensor(k, r) for r in results])


def switch(index, branches, *operands):
    """``branches[index](*operands)``, ``index`` (an int, or a traced int32
    scalar) clamped to the branches (``lax.switch``)."""
    branches = list(branches)
    if not isinstance(index, TracedTensor):
        return branches[min(max(int(index), 0), len(branches) - 1)](*operands)
    if len(branches) == 1:
        return branches[0](*operands)
    rest = lambda *o: switch(index - 1, branches[1:], *o)  # noqa: E731
    return cond(index <= 0, branches[0], rest, *operands)


def while_loop(cond_fun, body_fun, init_val):
    """``body_fun`` of the carried values (from ``init_val``) while
    ``cond_fun`` of them (a traced bool scalar) holds (``lax.while_loop``)."""
    k = _kernel()
    leaves, rebuild = _flatten(init_val)
    args = k.begin_while([v.var for v in leaves])
    c = cond_fun(rebuild([TracedTensor(k, a) for a in args]))
    args = k.begin_while_body(c.var)
    out, _ = _flatten(body_fun(rebuild([TracedTensor(k, a) for a in args])))
    results = k.end_while([v.var for v in out])
    return rebuild([TracedTensor(k, r) for r in results])


def scan(f, init, xs=None, length=None):
    """``lax.scan`` as Pallas lowers it on a TPU: of a carry alone, ``f``
    (``carry, None -> carry, ()``) run ``length`` times; the last carry and
    ``()``."""
    xs_leaves, rebuild = _flatten(xs)
    if length is None:
        if not xs_leaves:
            raise ValueError("scan needs a length, or xs to take it from")
        length = xs_leaves[0].shape[0]

    def body(_, carry):
        x = rebuild([prims.full(v.shape[1:], 0, v.dtype) for v in xs_leaves])
        carry, ys = f(carry, x)
        ys_leaves, _ = _flatten(ys)
        if ys_leaves:
            raise ValueError(f"Scan with {len(ys_leaves)} extensive output(s) is not supported")
        if xs_leaves:
            raise ValueError(f"Scan with {len(xs_leaves)} extensive argument(s) is not supported")
        return carry

    return fori_loop(0, length, body, init), ()


def loop(lower, upper, *, step=1, init_carry=None, unroll=None):
    """Runs the decorated ``body(i)`` (``body(i, carry)`` with
    ``init_carry``, giving the next) for ``i`` from ``lower`` below
    ``upper`` by ``step`` (``pl.loop``): the decorated name, the last carry."""
    del unroll

    def decorate(body):
        trips = (upper - lower + (step - 1 if step > 0 else step + 1)) // step
        if not isinstance(trips, TracedTensor):
            trips = max(trips, 0)
        if init_carry is None:
            fori_loop(0, trips, lambda j, c: body(_i32(lower) + j * step) and None, None)
            return None
        return fori_loop(0, trips, lambda j, c: body(_i32(lower) + j * step, c), init_carry)

    return decorate


class _Copy:
    def __init__(self, src, dst, sem):
        self.src, self.dst, self.sem = src, dst, sem

    def _issue(self, wait):
        indexed = [r for r in (self.src, self.dst) if isinstance(r, _Indexed)]
        if not indexed:
            self.src.kernel.dma(self.src.var, self.dst.var, self.sem.var, wait)
            return
        (rows,) = indexed
        src, dst = (r.ref if isinstance(r, _Indexed) else r for r in (self.src, self.dst))
        self.src.kernel.indirect_dma(src.var, dst.var, rows.offsets.var, self.sem.var, wait)

    def start(self):
        self._issue(False)

    def wait(self):
        self._issue(True)


def make_async_copy(src, dst, sem):
    """A copy of ref ``src`` to ``dst`` signalling ``sem``, to ``.start()``
    and ``.wait()`` (``pltpu.make_async_copy``)."""
    return _Copy(src, dst, sem)


def async_copy(src, dst, sem):
    """Starts copying ``src`` to ``dst``: the copy, to ``.wait()``
    (``pltpu.async_copy``)."""
    copy = _Copy(src, dst, sem)
    copy.start()
    return copy


def bitcast(x, dtype):
    """A value's or ref's bits as ``dtype``, of its width (``pltpu.bitcast``)."""
    if isinstance(x, Ref):
        return x.bitcast(dtype)
    return TracedTensor(x.graph, x.graph.bits("bitcast", [x.var], dtype=dtype))


def _bits(name, x, y):
    return TracedTensor(x.graph, x.graph.bits(name, [x.var, _like(x, y).var]))


def shift_left(x, y):
    return _bits("shift_left", x, y)


def shift_right_logical(x, y):
    return _bits("shift_right_logical", x, y)


def bitwise_and(x, y):
    return _bits("and", x, y)


def bitwise_or(x, y):
    return _bits("or", x, y)


def bitwise_xor(x, y):
    return prims.xor(x, _like(x, y))


def bitwise_not(x):
    return prims.not_(x)


def shift_right_arithmetic(x, y):
    return prims.shift_right_arithmetic(x, _like(x, y))


def integer_pow(x, y):
    """``x ** y``, ``y`` a Python int (``lax.integer_pow``): by squaring, as
    XLA, then its reciprocal if ``y`` is negative."""
    acc, n = None, abs(y)
    while n:
        if n & 1:
            acc = x if acc is None else acc * x
        n >>= 1
        if n:
            x = x * x
    if acc is None:
        return _like(x, 1)
    return _like(acc, 1) / acc if y < 0 else acc


def reduce_precision(x, exponent_bits, mantissa_bits):
    return prims.reduce_precision(x, exponent_bits, mantissa_bits)


def erf_inv(x, accurate=True):
    """``lax.erf_inv``: XLA's polynomial (accurate either way)."""
    del accurate
    return prims.erf_inv(x)


def delay(nanos):
    """Waits ``nanos`` (an int, or a traced int32 scalar) nanoseconds
    (``pltpu.delay``)."""
    _kernel().delay(_i32(nanos).var)


def population_count(x):
    return prims.population_count(x)


def clz(x):
    return prims.clz(x)


def broadcasted_iota(dtype, shape, dimension):
    return prims.iota(dtype, tuple(shape), dimension)


def select_n(which, *cases):
    """The case of index ``which`` (a bool: two cases; an int32), elementwise
    (``lax.select_n``)."""
    if not isinstance(which, TracedTensor):
        return cases[int(which)]
    if which.dtype == "bool":
        return prims.select(which, cases[1], cases[0])
    out = cases[0]
    for k, case in enumerate(cases[1:], 1):
        out = prims.select(which == k, case, out)
    return out


def dot_general(x, y, dimension_numbers, preferred_element_type=None):
    """``lax.dot_general``: ``dimension_numbers = ((lhs_contracting,
    rhs_contracting), (lhs_batch, rhs_batch))``, accumulated in float32
    (floats) or int32 (narrower integers if asked), returned in
    ``preferred_element_type`` (the operands' dtype by default)."""
    out = preferred_element_type or x.dtype
    accum = tracer._accum_dtype(x.dtype)
    if out == "int32" and x.dtype in ("int8", "int16", "uint8", "uint16"):
        accum = "int32"
    return prims.dot_general(x, y, dimension_numbers, accum, out)


def multiple_of(x, m):
    """``x`` (an int32 scalar), which the kernel promises is a multiple of
    ``m`` (``pl.multiple_of``)."""
    if not isinstance(x, TracedTensor):
        return x
    return TracedTensor(x.graph, x.graph.bits("multiple_of", [x.var], multiple=m))


def dot(a, b, trans_b=False, preferred_element_type=None):
    """``a @ b`` (``b.T`` if ``trans_b``) on the matrix unit, accumulated as
    ``@`` accumulates (floats in float32 or wider) and returned in
    ``preferred_element_type`` (the accumulator's by default); integers
    accumulate in ``preferred_element_type`` if given (int32 for the
    matrix unit)."""
    accum = tracer._accum_dtype(a.dtype)
    if preferred_element_type is not None and not tracer._is_float(a.dtype):
        accum = preferred_element_type
    numbers = (([1], [1 if trans_b else 0]), ([], []))
    return prims.dot_general(a, b, numbers, accum, preferred_element_type or accum)
