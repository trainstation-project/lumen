"""lumen's Pallas: TPU kernels written as Pallas kernels are, without JAX.

A kernel is a Python function of refs (buffers: HBM, VMEM, SMEM, DMA
semaphores), traced by lumen's tracer into a ``lumen._C.Kernel`` and
lowered to Mosaic in Rust (``lumen/pallas/mosaic.rs``)::

    from lumen import pallas as pl

    def body(x, y, o, acc):
        @pl.when(pl.program_id(2) == 0)
        def _():
            acc[...] = lumen.zeros(acc.shape)
        acc[...] += pl.dot(x[...], y[...])

    pl.emit_pipeline(body, grid=(mb, nb, kb),
                     in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)), ...],
                     out_specs=[...])(x_hbm, y_hbm, o_hbm)

Values in a kernel are lumen's traced tensors: lumen's ops (``@``,
``lumen.functional``, ``.to(dtype)``, indexing, ``lumen.zeros``) work on
them, lowered as Pallas lowers ``jax.numpy``'s. The API follows Pallas's
(``jax.experimental.pallas`` and its ``tpu``), whose design this ports
(Apache 2.0): ``ds``, ``program_id``, ``num_programs``, ``when``,
``fori_loop``, ``BlockSpec``, ``emit_pipeline``, ``async_copy``,
``make_async_copy``, ``VMEM``, ``SemaphoreType``, ``bitcast``,
``multiple_of`` and the TPU's bit ops.
"""

from lumen._C import Kernel
from lumen.graph import prims, tracer
from lumen.graph.tracer import TracedTensor

from .call import ANY, ShapeDtypeStruct, pallas_call, program
from .pipeline import BlockSpec, emit_pipeline, num_programs, program_id

__all__ = [
    "ANY",
    "BlockSpec",
    "Ref",
    "SemaphoreType",
    "ShapeDtypeStruct",
    "VMEM",
    "async_copy",
    "bitcast",
    "bitwise_and",
    "bitwise_or",
    "dot",
    "ds",
    "emit_pipeline",
    "fori_loop",
    "kernel",
    "make_async_copy",
    "multiple_of",
    "num_programs",
    "pallas_call",
    "program_id",
    "shift_left",
    "shift_right_logical",
    "when",
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
    int, or a traced int32 scalar) (``pl.ds``)."""

    def __init__(self, start, size):
        self.start, self.size = start, size


class VMEM:
    """A VMEM scratch buffer of ``shape`` and ``dtype`` (``pltpu.VMEM``)."""

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
        return self.ref._window(index)


class Ref:
    """A buffer of a kernel: its ``dtype``, ``shape`` and ``space`` (``"any"``
    for HBM, ``"vmem"``, ``"smem"``, ``"semaphore"``). ``ref[index]`` loads
    (a VMEM window whole, an SMEM element), ``ref[index] = value`` stores,
    ``ref.at[index]`` is a window of it (a ref), as in Pallas."""

    def __init__(self, kernel, var):
        self.kernel, self.var = kernel, var
        self.dtype, shape, self.space = kernel.ref_type(var)
        self.shape = tuple(shape)

    def __repr__(self):
        return f"Ref(%{self.var}, {self.dtype}{list(self.shape)}, {self.space})"

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def at(self):
        return _At(self)

    def _window(self, index):
        """The window of ``index``: ints and traced scalars pick an element
        (the dimension squeezed), slices and ``ds`` a range, ``...`` the
        rest."""
        index = index if isinstance(index, tuple) else (index,)
        if any(i is Ellipsis for i in index):
            at = index.index(Ellipsis)
            index = index[:at] + (slice(None),) * (self.ndim - len(index) + 1) + index[at + 1 :]
        index = index + (slice(None),) * (self.ndim - len(index))
        if len(index) != self.ndim:
            raise IndexError(f"{len(index)} indices for a ref of {self.ndim} dimensions")
        starts, dynamic, sizes, squeezed = [], [], [], []
        for i, n in zip(index, self.shape):
            if isinstance(i, slice):
                start, stop, step = i.indices(n)
                if step != 1:
                    raise IndexError("ref windows are contiguous")
                start, size, squeeze = start, stop - start, False
            elif isinstance(i, ds):
                start, size, squeeze = i.start, i.size, False
            else:
                start, size, squeeze = i, 1, True
            if isinstance(start, TracedTensor):
                starts.append(None)
                dynamic.append(start.var)
            else:
                starts.append(int(start))
            sizes.append(size)
            squeezed.append(squeeze)
        if starts == [0] * self.ndim and sizes == list(self.shape) and not any(squeezed):
            return self
        return Ref(self.kernel, self.kernel.slice(self.var, starts, dynamic, sizes, squeezed))

    def __getitem__(self, index):
        if self.space == "smem":
            return TracedTensor(self.kernel, self.kernel.load(self.var, _i32(index).var))
        return TracedTensor(self.kernel, self.kernel.load(self._window(index).var))

    def __setitem__(self, index, value):
        window = self._window(index)
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
        for s in scratch:
            var = k.vmem(s.dtype, list(s.shape)) if isinstance(s, VMEM) else k.semaphores(list(s.shape))
            refs.append(Ref(k, var))
        body(*refs)
    return k.lower(name)


def scratch(s):
    """A ref of scratch ``s`` (``VMEM``, ``SemaphoreType``) of the kernel
    being traced, made as it is traced (``emit_pipeline``'s buffers)."""
    k = _kernel()
    var = k.vmem(s.dtype, list(s.shape)) if isinstance(s, VMEM) else k.semaphores(list(s.shape))
    return Ref(k, var)


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


class _Copy:
    def __init__(self, src, dst, sem):
        self.src, self.dst, self.sem = src, dst, sem

    def start(self):
        self.src.kernel.dma(self.src.var, self.dst.var, self.sem.var, False)

    def wait(self):
        self.src.kernel.dma(self.src.var, self.dst.var, self.sem.var, True)


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


def multiple_of(x, m):
    """``x`` (an int32 scalar), which the kernel promises is a multiple of
    ``m`` (``pl.multiple_of``)."""
    if not isinstance(x, TracedTensor):
        return x
    return TracedTensor(x.graph, x.graph.bits("multiple_of", [x.var], multiple=m))


def dot(a, b, trans_b=False, preferred_element_type=None):
    """``a @ b`` (``b.T`` if ``trans_b``) on the matrix unit, accumulated as
    ``@`` accumulates (floats in float32 or wider) and returned in
    ``preferred_element_type`` (the accumulator's by default)."""
    accum = tracer._accum_dtype(a.dtype)
    numbers = (([1], [1 if trans_b else 0]), ([], []))
    return prims.dot_general(a, b, numbers, accum, preferred_element_type or accum)
