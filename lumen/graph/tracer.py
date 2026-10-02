"""Tracing, and the torch-like API on traced tensors.

``lumen.compile(fn)`` runs ``fn`` once per input signature with a
``TracedTensor`` standing in for each input tensor, recording the ops it
applies into a graph (JAX: a jaxpr), which then runs as a whole. Only
traced tensors have ops: a ``lumen.Tensor`` holds data, and the graph is
the only thing that computes with it.

The methods and functions here follow PyTorch (names, broadcasting, type
promotion, ``dim``/``keepdim``), and each is a composition of the strict
primitives in ``lumen.graph.prims``.
"""

import builtins
import functools
import math

from lumen._C import Graph, Plan, Tensor
from lumen.graph import prims
from lumen.tensor import default_dtype

__all__ = [
    "TracedTensor", "compile", "make_graph", "promote_types", "result_type",
    "where", "matmul", "maximum", "minimum", "exp", "log", "rsqrt", "tanh", "sigmoid", "softmax",
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
    call then runs. Results are on the first tensor argument's device."""
    plans = {}

    @functools.wraps(fn)
    def compiled(*args):
        key = _signature(args)
        if key not in plans:
            graph, single = _trace(fn, args)
            device = next((a.device for a in args if isinstance(a, Tensor)), None)
            plans[key] = Plan(graph, device), single
        plan, single = plans[key]
        outputs = plan.run([a for a in args if isinstance(a, Tensor)])
        return outputs[0] if single else tuple(outputs)

    return compiled


def make_graph(fn):
    """A function returning the graph ``fn`` traces to on the given
    arguments (``jax.make_jaxpr``); ``print`` it to read it."""

    @functools.wraps(fn)
    def graph(*args):
        return _trace(fn, args)[0]

    return graph


# ---------------------------------------------------------------------
# dtypes (torch.promote_types, torch.result_type)
# ---------------------------------------------------------------------


def _kind(dtype):
    return "b" if dtype == "bool" else "f" if "float" in dtype else "u" if dtype.startswith("u") else "i"


def _bits(dtype):
    return int(dtype.lstrip("bfloatuint"))


def _is_float(dtype):
    return _kind(dtype) == "f"


def promote_types(a, b):
    """The smallest dtype both ``a`` and ``b`` convert to without loss
    (``torch.promote_types``)."""
    if a == b:
        return a
    ka, kb = _kind(a), _kind(b)
    if ka == "b" or kb == "b":
        return b if ka == "b" else a
    if ka != kb and "f" in (ka, kb):
        return a if ka == "f" else b
    if {a, b} == {"float16", "bfloat16"}:
        return "float32"
    if ka == kb:
        return builtins.max(a, b, key=_bits)
    signed, unsigned = (a, b) if ka == "i" else (b, a)
    if _bits(signed) > _bits(unsigned):
        return signed
    if _bits(unsigned) == 64:
        raise TypeError(f"promotion of {unsigned} with {signed} is not supported")
    return f"int{2 * _bits(unsigned)}"


def _scalar_dtype(x):
    return "bool" if isinstance(x, bool) else "int64" if isinstance(x, int) else default_dtype


def _category(dtype):
    return {"b": 0, "u": 1, "i": 1, "f": 2}[_kind(dtype)]


def result_type(*operands):
    """The dtype an op computes in (``torch.result_type``): tensors with
    dimensions decide it, then zero-dimensional tensors, then Python
    scalars; a later group only matters if it is of a higher category
    (bool < integer < float), and a Python float then means the default
    dtype."""
    groups = [
        [x.dtype for x in operands if isinstance(x, TracedTensor) and x.ndim],
        [x.dtype for x in operands if isinstance(x, TracedTensor) and not x.ndim],
        [_scalar_dtype(x) for x in operands if not isinstance(x, TracedTensor)],
    ]
    result = None
    for dtypes in groups:
        if dtypes:
            dtype = functools.reduce(promote_types, dtypes)
            if result is None or _category(dtype) > _category(result):
                result = dtype
    return result


# ---------------------------------------------------------------------
# broadcasting and dimensions
# ---------------------------------------------------------------------


def _is_operand(x):
    return isinstance(x, (TracedTensor, bool, int, float))


def _as_tensor(x, dtype):
    """``x`` (a traced tensor or Python scalar) as a traced tensor of ``dtype``."""
    if isinstance(x, TracedTensor):
        return x.to(dtype)
    if not _is_operand(x):
        raise TypeError(f"expected a tensor or a Python scalar, got {type(x).__name__}")
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


def _elementwise(prim, *operands, dtype=None):
    """``prim`` on ``operands`` promoted to ``dtype`` (default: their result
    type) and broadcast to a common shape."""
    dtype = dtype or result_type(*operands)
    tensors = [_as_tensor(x, dtype) for x in operands]
    shape = _broadcast_shapes(*(t.shape for t in tensors))
    return prim(*(_broadcast_to(t, shape) for t in tensors))


def _to_float(x):
    """Integer and bool tensors to the default dtype, as torch does for
    floating-point functions."""
    return x if _is_float(x.dtype) else x.to(default_dtype)


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
    dtype = result_type(x, y)
    return _elementwise(prims.div, x, y, dtype=dtype if _is_float(dtype) else default_dtype)


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
        return prims.exp(_to_float(self))

    def log(self):
        return prims.log(_to_float(self))

    def rsqrt(self):
        return prims.rsqrt(_to_float(self))

    def tanh(self):
        return prims.tanh(_to_float(self))

    def sigmoid(self):
        return prims.logistic(_to_float(self))

    def relu(self):
        return _elementwise(prims.max, self, 0, dtype=self.dtype)

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
        """Sum over ``dim`` (all dimensions if None); integers and bools sum
        to int64, as in torch."""
        x = self.to(dtype or (self.dtype if _is_float(self.dtype) else "int64"))
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
        x = self.to(dtype) if dtype else _to_float(self)
        e = (x - x.amax(dim, keepdim=True)).exp()
        return e / e.sum(dim, keepdim=True)

    def log_softmax(self, dim, dtype=None):
        x = self.to(dtype) if dtype else _to_float(self)
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
    dtype = result_type(input, other)
    input, other = _as_tensor(input, dtype), _as_tensor(other, dtype)
    shape = _broadcast_shapes(condition.shape, input.shape, other.shape)
    return prims.select(*(_broadcast_to(t, shape) for t in (condition, input, other)))


def matmul(input, other):
    """``input @ other`` with torch's rules: 1-d operands are vectors, and
    the dimensions before the last two are batch dimensions, broadcast."""
    if input.dtype != other.dtype:
        raise RuntimeError(f"expected both matmul operands to have the same dtype, but got {input.dtype} and {other.dtype}")
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
