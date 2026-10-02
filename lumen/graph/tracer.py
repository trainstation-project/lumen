"""Tracing, and the torch-like API on traced tensors.

``lumen.compile(fn)`` runs ``fn`` once per input signature with a
``TracedTensor`` standing in for each input tensor, recording the ops it
applies into a graph (JAX: a jaxpr), which then runs as a whole. Only
traced tensors have ops: a ``lumen.Tensor`` holds data, and the graph is
the only thing that computes with it.

The methods and functions here follow PyTorch (names, broadcasting,
``dim``/``keepdim``), and each is a composition of the strict primitives in
``lumen.graph.prims``. Unlike PyTorch, no op changes a dtype the user did
not ask for: operands must share a dtype (a Python scalar takes its tensor
operand's, and must be of its kind), and floating-point functions take
floating-point tensors. Convert with ``.to(dtype)``, or an op's ``dtype=``.
"""

import builtins
import functools
import math

from lumen._C import Graph, Plan, Tensor
from lumen.graph import prims

__all__ = [
    "TracedTensor", "compile", "make_graph", "where", "matmul", "maximum", "minimum", "exp", "log", "rsqrt", "tanh", "sigmoid", "softmax",
]

# The graphs being traced, innermost last.
_TRACES = []


def current_graph():
    if not _TRACES:
        raise RuntimeError("lumen ops run only while tracing, inside a function passed to lumen.compile")
    return _TRACES[-1]


# ---------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------


def _trace(fn, args):
    """``fn`` traced on ``args``: the graph, and whether ``fn`` returned a
    single tensor (rather than a tuple or list of them)."""
    graph = Graph()
    traced = [TracedTensor(graph, graph.input(a.dtype, a.shape)) if isinstance(a, Tensor) else a for a in args]
    _TRACES.append(graph)
    try:
        out = fn(*traced)
    finally:
        _TRACES.pop()
    single = isinstance(out, TracedTensor)
    outputs = (out,) if single else tuple(out)
    for o in outputs:
        if not isinstance(o, TracedTensor) or o.graph is not graph:
            raise TypeError(f"a compiled function must return tensors computed from its inputs, got {o!r}")
    graph.set_outputs([o.var for o in outputs])
    return graph, single


def _signature(args):
    # Tensors are traced by dtype and shape, and compiled for their device;
    # anything else is baked into the graph, so it must be hashable.
    return tuple((a.dtype, tuple(a.shape), str(a.device)) if isinstance(a, Tensor) else ("static", a) for a in args)


def compile(fn):
    """``fn`` compiled (``torch.compile``, ``jax.jit``): traced into a graph
    and compiled into a static plan for the tensor arguments' device on its
    first call with each input signature (the tensor arguments' dtypes,
    shapes and devices, and the values of the other arguments), which every
    call then runs. Results are on the first tensor argument's device.

    ``compiled.dump_graph(path)`` writes the graph and plan of the latest
    call's signature, profiled, as an HTML page (``lumen.graph.viz``);
    ``compiled.dump_graph(path, *args)``, those of ``args``' signature."""
    plans = {}
    latest = []

    def entry(args):
        key = _signature(args)
        if key not in plans:
            graph, single = _trace(fn, args)
            device = next((a.device for a in args if isinstance(a, Tensor)), None)
            plans[key] = graph, Plan(graph, device), single
        latest[:] = [key]
        return plans[key]

    @functools.wraps(fn)
    def compiled(*args):
        _, plan, single = entry(args)
        outputs = plan.run([a for a in args if isinstance(a, Tensor)])
        return outputs[0] if single else tuple(outputs)

    def dump_graph(path, *args, runs=5, json_path=None, fragment=False):
        """Write the graph and plan for ``args``' signature (default: the
        latest call's) to ``path`` as an HTML page, profiled over ``runs``
        runs on new tensors of the signature's types; ``json_path`` also
        gets the page's data, which is returned. ``fragment`` leaves out the
        doctype, for a host that wraps the page (a published artifact)."""
        if args:
            entry(args)
        elif not latest:
            raise RuntimeError(f"dump_graph: call {fn.__name__} first, or pass it arguments to trace")
        key = latest[0]
        graph, plan, _ = plans[key]
        # Kernels do not depend on the values: profile on ones.
        inputs = [Tensor.ones(list(shape), dtype, device) for dtype, shape, device in (k for k in key if k[0] != "static")]
        from lumen.graph import viz

        data = viz.collect(graph, plan, inputs, title=fn.__name__, runs=runs)
        viz.write(data, path, json_path=json_path, fragment=fragment)
        return data

    compiled.dump_graph = dump_graph
    return compiled


def make_graph(fn):
    """A function returning the graph ``fn`` traces to on the given
    arguments (``jax.make_jaxpr``); ``print`` it to read it."""

    @functools.wraps(fn)
    def graph(*args):
        return _trace(fn, args)[0]

    return graph


# ---------------------------------------------------------------------
# dtypes: never changed implicitly
# ---------------------------------------------------------------------


def _is_float(dtype):
    return "float" in dtype


def _require_float(x, op):
    if not _is_float(x.dtype):
        raise TypeError(f"{op} needs a floating-point tensor, got {x.dtype}: convert it with .to(dtype)")
    return x


def _common_dtype(op, operands):
    """The dtype the tensors among ``operands`` share."""
    dtypes = sorted({x.dtype for x in operands if isinstance(x, TracedTensor)})
    if not dtypes:
        raise TypeError(f"{op} needs a tensor operand")
    if len(dtypes) > 1:
        raise TypeError(f"{op} got tensors of dtypes {' and '.join(dtypes)}: convert them to one with .to(dtype)")
    return dtypes[0]


def _scalar_fits(x, dtype):
    """Whether Python scalar ``x`` is of ``dtype``'s kind: a bool of bool, an
    int of an integer or floating-point dtype, a float of a floating-point
    one."""
    if isinstance(x, bool):
        return dtype == "bool"
    if isinstance(x, int):
        return dtype != "bool"
    return _is_float(dtype)


# ---------------------------------------------------------------------
# broadcasting and dimensions
# ---------------------------------------------------------------------


def _is_operand(x):
    return isinstance(x, (TracedTensor, bool, int, float))


def _as_tensor(x, dtype, op):
    """``x`` (a traced tensor of ``dtype``, or a Python scalar of its kind)
    as a traced tensor."""
    if isinstance(x, TracedTensor):
        return x
    if not _is_operand(x):
        raise TypeError(f"expected a tensor or a Python scalar, got {type(x).__name__}")
    if not _scalar_fits(x, dtype):
        raise TypeError(f"{op} of a {dtype} tensor and the {type(x).__name__} {x!r}: convert the tensor with .to(dtype)")
    return prims.full((), x, dtype)


def _broadcast_shapes(*shapes):
    ndim = builtins.max(len(s) for s in shapes)
    out = []
    for d in range(-ndim, 0):
        sizes = {s[d] for s in shapes if len(s) >= -d} - {1}
        if len(sizes) > 1:
            raise RuntimeError(f"shapes {', '.join(map(str, map(list, shapes)))} are not broadcastable at dimension {d}")
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


def _broadcast_to(x, shape):
    if x.shape == shape:
        return x
    lead = len(shape) - x.ndim
    return prims.broadcast_in_dim(x, shape, range(lead, len(shape)))


def _operands(op, *operands):
    """``operands`` as traced tensors of their common dtype, broadcast to a
    common shape."""
    dtype = _common_dtype(op, operands)
    tensors = [_as_tensor(x, dtype, op) for x in operands]
    shape = _broadcast_shapes(*(t.shape for t in tensors))
    return [_broadcast_to(t, shape) for t in tensors]


def _elementwise(prim, *operands):
    """``prim`` on ``operands``, of one dtype, broadcast to a common shape."""
    return prim(*_operands(prim.__name__.lstrip("_"), *operands))


def _dim(dim, ndim):
    """``dim`` in ``[-ndim, ndim)`` as a nonnegative dimension; a 0-d tensor
    takes dimension 0 or -1."""
    n = builtins.max(ndim, 1)
    if not -n <= dim < n:
        raise IndexError(f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {dim})")
    return dim % n


def _dims(dim, ndim):
    """A ``dim=`` argument (None, an int, or a sequence) as a sorted tuple;
    None or ``()`` means every dimension."""
    if dim is None or dim == ():
        return tuple(range(ndim))
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    return tuple(sorted({_dim(d, ndim) for d in dims} & set(range(ndim))))


def _sizes(args):
    """``*sizes`` or a single sequence of sizes, as ``reshape`` and friends take."""
    return tuple(args[0]) if len(args) == 1 and not isinstance(args[0], int) else tuple(args)


def _min(x, y):
    # -max(-x, -y) keeps NaN propagation for floats; integers select.
    if _is_float(x.dtype):
        return prims.neg(prims.max(prims.neg(x), prims.neg(y)))
    return prims.select(prims.lt(y, x), y, x)


def _le(x, y):
    less = prims.lt(x, y)
    return prims.select(less, less, prims.eq(x, y))


def _ne(x, y):
    return prims.eq(prims.eq(x, y), prims.full(x.shape, False, "bool"))


# ---------------------------------------------------------------------
# TracedTensor
# ---------------------------------------------------------------------


def _binary(prim, reflected=False):
    """An operator method: ``prim`` on the operands, promoted and broadcast."""

    def op(self, other):
        if not _is_operand(other):
            return NotImplemented
        return _elementwise(prim, other, self) if reflected else _elementwise(prim, self, other)

    return op


def _true_div(x, y):
    x, y = _operands("div", x, y)
    return prims.div(_require_float(x, "true division (/)"), y)


class TracedTensor:
    """A value of the graph being traced, standing in for a tensor inside a
    function passed to ``lumen.compile``. It has a dtype and shape but no
    data; its methods record primitives into the graph."""

    __slots__ = ("graph", "var", "dtype", "shape")

    def __init__(self, graph, var):
        self.graph = graph
        self.var = var
        dtype, shape = graph.type_of(var)
        self.dtype = dtype
        self.shape = tuple(shape)

    def __repr__(self):
        return f"TracedTensor(%{self.var}, dtype={self.dtype}, shape={list(self.shape)})"

    def __bool__(self):
        raise TypeError("a traced tensor has no value: control flow cannot depend on tensor data inside lumen.compile")

    __hash__ = object.__hash__

    # -- metadata -------------------------------------------------------

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[_dim(dim, self.ndim)]

    def numel(self):
        return math.prod(self.shape)

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    # -- arithmetic and comparison ----------------------------------------

    __add__ = _binary(prims.add)
    __radd__ = _binary(prims.add, reflected=True)
    __sub__ = _binary(prims.sub)
    __rsub__ = _binary(prims.sub, reflected=True)
    __mul__ = _binary(prims.mul)
    __rmul__ = _binary(prims.mul, reflected=True)
    __eq__ = _binary(prims.eq)
    __ne__ = _binary(_ne)
    __lt__ = _binary(prims.lt)
    __le__ = _binary(_le)
    __gt__ = _binary(prims.lt, reflected=True)
    __ge__ = _binary(_le, reflected=True)

    def __truediv__(self, other):
        return _true_div(self, other) if _is_operand(other) else NotImplemented

    def __rtruediv__(self, other):
        return _true_div(other, self) if _is_operand(other) else NotImplemented

    def __neg__(self):
        return prims.neg(self)

    def __matmul__(self, other):
        return matmul(self, other) if isinstance(other, TracedTensor) else NotImplemented

    def __rmatmul__(self, other):
        return matmul(other, self) if isinstance(other, TracedTensor) else NotImplemented

    def add(self, other):
        return self + other

    def sub(self, other):
        return self - other

    def mul(self, other):
        return self * other

    def div(self, other):
        return self / other

    def neg(self):
        return -self

    def eq(self, other):
        return self == other

    def ne(self, other):
        return self != other

    def lt(self, other):
        return self < other

    def le(self, other):
        return self <= other

    def gt(self, other):
        return self > other

    def ge(self, other):
        return self >= other

    def maximum(self, other):
        return maximum(self, other)

    def minimum(self, other):
        return minimum(self, other)

    # -- elementwise functions -------------------------------------------

    def exp(self):
        return prims.exp(_require_float(self, "exp"))

    def log(self):
        return prims.log(_require_float(self, "log"))

    def rsqrt(self):
        return prims.rsqrt(_require_float(self, "rsqrt"))

    def tanh(self):
        return prims.tanh(_require_float(self, "tanh"))

    def sigmoid(self):
        return prims.logistic(_require_float(self, "sigmoid"))

    def relu(self):
        return _elementwise(prims.max, self, 0)

    # -- dtype conversion --------------------------------------------------

    def to(self, dtype):
        return self if dtype == self.dtype else prims.convert_element_type(self, dtype)

    def type_as(self, other):
        return self.to(other.dtype)

    def float(self):
        return self.to("float32")

    def double(self):
        return self.to("float64")

    def half(self):
        return self.to("float16")

    def bfloat16(self):
        return self.to("bfloat16")

    def int(self):
        return self.to("int32")

    def long(self):
        return self.to("int64")

    def bool(self):
        return self.to("bool")

    # -- reductions ----------------------------------------------------------

    def _keep(self, out, dims, keepdim):
        if not keepdim:
            return out
        return out.reshape(tuple(1 if d in dims else n for d, n in enumerate(self.shape)))

    def sum(self, dim=None, keepdim=False, dtype=None):
        """Sum over ``dim`` (all dimensions if None), in the tensor's dtype
        (integers wrap) or ``dtype``."""
        x = self.to(dtype) if dtype else self
        dims = _dims(dim, self.ndim)
        return self._keep(prims.reduce_sum(x, dims), dims, keepdim)

    def mean(self, dim=None, keepdim=False, dtype=None):
        x = self.to(dtype) if dtype else self
        if not _is_float(x.dtype):
            raise RuntimeError(f"mean(): input dtype must be floating point, got {x.dtype}")
        dims = _dims(dim, self.ndim)
        count = math.prod(self.shape[d] for d in dims)
        return x.sum(dims, keepdim) / count

    def amax(self, dim=(), keepdim=False):
        dims = _dims(dim, self.ndim)
        return self._keep(prims.reduce_max(self, dims), dims, keepdim)

    def max(self, other=None, keepdim=False):
        """``max()`` over every element, or the elementwise ``max(other)``.
        ``max(dim)``, which also returns indices, is not supported: use
        ``amax(dim)``."""
        if other is None:
            return self.amax()
        if isinstance(other, TracedTensor):
            return maximum(self, other)
        raise NotImplementedError("max(dim) returns indices, which lumen does not support yet; use amax(dim)")

    def softmax(self, dim, dtype=None):
        x = _require_float(self.to(dtype) if dtype else self, "softmax")
        e = (x - x.amax(dim, keepdim=True)).exp()
        return e / e.sum(dim, keepdim=True)

    def log_softmax(self, dim, dtype=None):
        x = _require_float(self.to(dtype) if dtype else self, "log_softmax")
        shifted = x - x.amax(dim, keepdim=True)
        return shifted - shifted.exp().sum(dim, keepdim=True).log()

    # -- shape -------------------------------------------------------------

    def reshape(self, *shape):
        shape = list(_sizes(shape))
        if shape.count(-1) > 1:
            raise RuntimeError("only one dimension can be inferred")
        if -1 in shape:
            known = math.prod(n for n in shape if n != -1)
            if known == 0 or self.numel() % known:
                raise RuntimeError(f"shape '{shape}' is invalid for input of size {self.numel()}")
            shape[shape.index(-1)] = self.numel() // known
        if math.prod(shape) != self.numel():
            raise RuntimeError(f"shape '{shape}' is invalid for input of size {self.numel()}")
        return self if tuple(shape) == self.shape else prims.reshape(self, shape)

    view = reshape

    def flatten(self, start_dim=0, end_dim=-1):
        if self.ndim == 0:
            return self.reshape(1)
        start, end = _dim(start_dim, self.ndim), _dim(end_dim, self.ndim)
        return self.reshape(self.shape[:start] + (-1,) + self.shape[end + 1:])

    def unsqueeze(self, dim):
        dim = _dim(dim, self.ndim + 1)
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim=None):
        dims = _dims(dim, self.ndim)
        return self.reshape(tuple(n for d, n in enumerate(self.shape) if not (d in dims and n == 1)))

    def permute(self, *dims):
        dims = [_dim(d, self.ndim) for d in _sizes(dims)]
        if sorted(dims) != list(range(self.ndim)):
            raise RuntimeError(f"permute: {dims} is not a permutation of the dimensions of a {self.ndim}-d tensor")
        return self if dims == sorted(dims) else prims.transpose(self, dims)

    def transpose(self, dim0, dim1):
        dims = list(range(self.ndim))
        a, b = _dim(dim0, self.ndim), _dim(dim1, self.ndim)
        dims[a], dims[b] = dims[b], dims[a]
        return self.permute(dims)

    def t(self):
        if self.ndim > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {self.ndim}D")
        return self.transpose(0, -1) if self.ndim == 2 else self

    @property
    def mT(self):
        return self.transpose(-2, -1)

    def expand(self, *sizes):
        sizes = _sizes(sizes)
        lead = len(sizes) - self.ndim
        if lead < 0:
            raise RuntimeError(f"expand: the number of sizes ({len(sizes)}) must be at least the tensor's rank ({self.ndim})")
        shape = tuple(self.shape[i - lead] if n == -1 and i >= lead else n for i, n in enumerate(sizes))
        return _broadcast_to(self, shape)

    def expand_as(self, other):
        return self.expand(other.shape)

    def broadcast_to(self, shape):
        return self.expand(shape)

    def matmul(self, other):
        return matmul(self, other)


# ---------------------------------------------------------------------
# functions (torch.where, torch.matmul, ...)
# ---------------------------------------------------------------------


def where(condition, input, other):
    if not isinstance(condition, TracedTensor) or condition.dtype != "bool":
        raise TypeError("where expected condition to be a bool tensor")
    dtype = _common_dtype("where", (input, other))
    input, other = _as_tensor(input, dtype, "where"), _as_tensor(other, dtype, "where")
    shape = _broadcast_shapes(condition.shape, input.shape, other.shape)
    return prims.select(*(_broadcast_to(t, shape) for t in (condition, input, other)))


def matmul(input, other):
    """``input @ other`` with torch's rules: 1-d operands are vectors, and
    the dimensions before the last two are batch dimensions, broadcast."""
    _common_dtype("matmul", (input, other))
    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D")
    x = input.unsqueeze(0) if input.ndim == 1 else input
    y = other.unsqueeze(-1) if other.ndim == 1 else other
    batch = _broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x, y = _broadcast_to(x, batch + x.shape[-2:]), _broadcast_to(y, batch + y.shape[-2:])
    b = tuple(range(len(batch)))
    out = prims.dot_general(x, y, (((len(b) + 1,), (len(b),)), (b, b)))
    if input.ndim == 1:
        out = out.squeeze(-2)
    if other.ndim == 1:
        out = out.squeeze(-1)
    return out


def maximum(input, other):
    return _elementwise(prims.max, input, other)


def minimum(input, other):
    return _elementwise(_min, input, other)


def exp(input):
    return input.exp()


def log(input):
    return input.log()


def rsqrt(input):
    return input.rsqrt()


def tanh(input):
    return input.tanh()


def sigmoid(input):
    return input.sigmoid()


def softmax(input, dim, dtype=None):
    return input.softmax(dim, dtype)
