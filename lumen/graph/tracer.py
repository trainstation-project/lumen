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

from lumen._C import Graph, Plan, Tensor, _pack
from lumen import nn
from lumen.graph import prims

__all__ = [
    "TracedTensor", "compile", "make_graph", "where", "matmul", "maximum", "minimum", "exp", "log", "sqrt", "tanh", "sigmoid", "softmax", "rms_norm",
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


def _weights(args):
    """The weights of ``args``' modules (``lumen.nn.Module``), each tensor
    once, in order: whole meta tensors."""
    params = {}
    for a in args:
        if isinstance(a, nn.Module):
            for _, t in nn._leaves(a, ""):
                if not t._is_parameter:
                    raise TypeError(f"a module's weights must be whole meta tensors, got a {t.device} tensor of shape {t.shape}")
                params.setdefault(t.storage_id, t)
    return list(params.values())


def _scalars(args):
    """The runtime scalars of ``args``: the float arguments, then the float
    fields of the module arguments, in order."""
    floats = [a for a in args if isinstance(a, float)]
    return floats + [v for a in args if isinstance(a, nn.Module) for v in nn._floats(a)]


def _trace(fn, args):
    """``fn`` traced on ``args``: the graph, and whether ``fn`` returned a
    single tensor (rather than a tuple or list of them). The graph's inputs
    are the tensor arguments, then the runtime scalars (``_scalars``: 0-d
    float32 inputs, weakly typed: each takes its tensor operand's dtype),
    then the weights of the module arguments (``_weights``), each a whole
    meta tensor."""
    graph = Graph()
    traced = [TracedTensor(graph, graph.input(a.dtype, a.shape)) if isinstance(a, Tensor) else a for a in args]

    def scalar(_):
        t = TracedTensor(graph, graph.input("float32", []))
        t.weak = True
        return t

    traced = [scalar(a) if isinstance(a, float) else a for a in traced]
    scalars = [scalar(v) for a in args if isinstance(a, nn.Module) for v in nn._floats(a)]
    weights = {}
    for t in _weights(args):
        weights[t.storage_id] = TracedTensor(graph, graph.input(t.dtype, t.shape))
    module_scalars = iter(scalars)
    traced = [
        nn.map_tensors(a, lambda t: weights[t.storage_id], lambda _: next(module_scalars))
        if isinstance(a, nn.Module)
        else a
        for a in traced
    ]
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


def _lift(x):
    """``x`` as an operand of the graph being traced; a tensor the function
    closes over is an error (it would be invisible to the graph)."""
    if isinstance(x, Tensor) and _TRACES:
        raise TypeError(
            f"a compiled function cannot close over a tensor ({x.device}, shape {x.shape}): pass data as an argument, and weights in a lumen.nn.Module argument"
        )
    return x


def _signature(args):
    # Tensors are traced by dtype and shape (data or meta alike), floats as
    # runtime scalars (their values change nothing), modules by their
    # structure and which of their weights are one tensor; anything else is
    # baked into the graph, so it must be hashable.
    ids = [t.storage_id for a in args if isinstance(a, nn.Module) for _, t in nn._leaves(a, "")]
    shared = tuple(ids.index(i) for i in ids)
    return tuple(
        (a.dtype, tuple(a.shape))
        if isinstance(a, Tensor)
        else nn.structure(a)
        if isinstance(a, (nn.Module, float))
        else ("static", a)
        for a in args
    ) + (shared,)


def _device(args, device):
    """Where a function of ``args`` runs: ``device``, or the first tensor
    argument with data's, or the meta device if all are meta (nothing
    runs), or the CPU."""
    if device is not None:
        return str(device)
    tensors = [a for a in args if isinstance(a, Tensor)]
    data = [a for a in tensors if a.device != "meta"]
    return str(data[0].device) if data else ("meta" if tensors else "cpu")


def compile(fn, device=None):
    """``fn`` compiled (``torch.compile``, ``jax.jit``): traced into a graph
    and compiled into a static plan for ``device`` on its first call with
    each input signature (the tensor arguments' dtypes and shapes, and the
    values of the other arguments but floats), which every call then runs.
    Float arguments, and float fields of module arguments, are runtime
    scalars, as in ``jax.jit``: inputs of the plan, whose values each call
    passes (another value reuses the plan), weakly typed (each takes its
    tensor operand's dtype). Ints and the rest are static. ``device``
    defaults to the first tensor argument with data's (the CPU without
    any).

    The compiled function owns its memory (XLA: buffer assignment over the
    whole program). Its plan places every value, its tensor arguments and
    results included, in one workspace, allocated once: each call copies
    the arguments in (from any device) and returns views of the results,
    which the next call overwrites (``.clone()`` one to keep it).

    Its weights are the tensors of its ``lumen.nn.Module`` arguments (meta
    tensors): never the caller's to allocate, they are placed on the device
    when the first compiled function taking them compiles, and every
    compiled function taking them shares that memory, read in place (a
    module argument costs no copy). Where dots that share an operand merge
    into one (``x @ w1`` and ``x @ w3``), their weights are placed side by side
    in one block, read by the merged dot, never concatenated. A call with
    meta tensor arguments compiles without running (meta results); then
    ``model.load_state_dict`` (or ``w.copy_``) writes the weights' data into
    their memory, once::

        f = lumen.compile(lambda model, x: model(x), device="mps")
        f(model, lumen.empty([seq, dim], device="meta"))  # compile: place weights
        model.load_state_dict(lumen.safetensors.load_file("model.safetensors"))
        y = f(model, x)  # each call copies x in

    With only meta tensors and no ``device``, nothing is compiled for a
    device: results are meta tensors of the right types.

    ``compiled.dump_graph(path)`` writes the graph and plan of the latest
    call's signature, profiled, as an HTML page (``lumen.graph.viz``);
    ``compiled.dump_graph(path, *args)``, those of ``args``' signature."""
    compile_device = device  # dump_graph's own `device` shadows it
    # Per signature and device: the graph, whether `fn` returns one tensor,
    # its plans, each with its workspace (`None` on the meta device), tried
    # in order, and the number of tensor arguments (the graph's inputs
    # before the weights).
    plans = {}
    latest = []

    def prepare(args):
        """The plan for ``args`` and its inputs: the graph, the plan, its
        workspace, whether ``fn`` returns a single tensor, and the plan's
        inputs (the arguments, then the weights placed, and their blocks)."""
        target = _device(args, device)
        key = _signature(args), target
        # The arguments the plan copies in: tensors, then runtime scalars.
        # The kernels take the scalars by value, read from the host on each
        # call: a new value needs no new plan.
        tensors = [a for a in args if isinstance(a, Tensor)]
        scalars = list(range(len(tensors), len(tensors) + len(_scalars(args))))
        tensors += [Tensor.full([], v, "float32") for v in _scalars(args)]
        if key not in plans:
            plans[key] = (*_trace(fn, args), [], len(tensors), scalars)
        graph, single, entries, _, _ = plans[key]
        weights = _weights(args)
        latest[:] = [key]
        if target == "meta":
            if not entries:
                entries.append((Plan(graph, "meta"), None))
            return graph, entries[0][0], None, single, tensors + weights
        params = list(range(len(tensors), len(tensors) + len(weights)))
        inputs = tensors + weights

        def inputs_for(plan):
            # The weights in their blocks, if they are placed so (or not
            # yet placed), else None.
            blocks = []
            for positions, dimension in plan.packed:
                block = _pack([inputs[i] for i in positions], dimension, target)
                if block is None:
                    return None
                blocks.append(block)
            return tensors + [w._placed(target) for w in weights] + blocks

        for plan, workspace in entries:
            placed = inputs_for(plan)
            if placed is not None:
                return graph, plan, workspace, single, placed
        # A new plan: dots merged into blocks of weights while none of theirs
        # is placed yet; once they are placed otherwise, without.
        packable = params if not entries else []
        plan = Plan(graph, target, parameters=params, packable=packable, scalars=scalars)
        workspace = Tensor.empty([plan.workspace_bytes], "uint8", target)
        entries.append((plan, workspace))
        placed = inputs_for(plan)
        if placed is None:
            plan = Plan(graph, target, parameters=params, scalars=scalars)
            entries[-1] = plan, workspace
            placed = inputs_for(plan)
        return graph, plan, workspace, single, placed

    @functools.wraps(fn)
    def compiled(*args):
        graph, plan, workspace, single, inputs = prepare(args)
        if workspace is None or any(isinstance(a, Tensor) and a.device == "meta" for a in args):
            # Compiled (and the weights placed); nothing to run.
            outputs = [Tensor.empty(list(shape), dtype, "meta") for dtype, shape in map(graph.type_of, graph.outputs())]
        else:
            outputs = plan.run_in(workspace, inputs)
        return outputs[0] if single else tuple(outputs)

    def dump_graph(path, *args, device=None, runs=5, json_path=None, fragment=False):
        """Write the graph and plan for ``args``' signature (default: the
        latest call's; meta tensors trace it without data) to ``path`` as an
        HTML page, compiled for and profiled on ``device`` (default: the
        function's) over ``runs`` runs on new tensors of the signature's
        types; ``json_path`` also gets the page's data, which is returned.
        ``fragment`` leaves out the doctype, for a host that wraps the page
        (a published artifact)."""
        if args:
            prepare(args)
        elif not latest:
            raise RuntimeError(f"dump_graph: call {fn.__name__} first, or pass it arguments to trace")
        ((_, target),) = latest
        graph, _, _, n_tensors, scalars = plans[latest[0]]
        target = str(device or compile_device or (target if target != "meta" else "cpu"))
        # Kernels do not depend on the values: profile on ones, the weights
        # packed where the compiler merges dots.
        types = [graph.type_of(v) for v in graph.inputs()]
        params = list(range(n_tensors, len(types)))
        plan = Plan(graph, target, parameters=params, packable=params, scalars=scalars)
        inputs = [Tensor.ones(list(shape), dtype, target) for dtype, shape in types]
        for positions, dimension in plan.packed:
            dtype, shape = types[positions[0]]
            shape = list(shape)
            shape[dimension] = sum(types[i][1][dimension] for i in positions)
            inputs.append(Tensor.ones(shape, dtype, target))
        workspace = Tensor.empty([plan.workspace_bytes], "uint8", target)
        from lumen.graph import viz

        data = viz.collect(
            graph, plan, inputs, title=fn.__name__, runs=runs, device=target, run=lambda: plan.run_in(workspace, inputs)
        )
        viz.write(data, path, json_path=json_path, fragment=fragment)
        return data

    compiled.dump_graph = dump_graph
    return compiled


def make_graph(fn):
    """A function returning the graph ``fn`` traces to on the given
    arguments (``jax.make_jaxpr``), which may be meta tensors; ``print`` it
    to read it."""

    @functools.wraps(fn)
    def graph(*args):
        return _trace(fn, args)[0]

    return graph


# ---------------------------------------------------------------------
# dtypes: never changed implicitly
# ---------------------------------------------------------------------


def _is_float(dtype):
    return "float" in dtype


def _accum_dtype(dtype):
    """The dtype sums of ``dtype`` accumulate in (matmul, sum, mean), their
    result's: float32 for floats, or their own if wider (float64); integers
    their own."""
    return dtype if not _is_float(dtype) or dtype == "float64" else "float32"


def _require_float(x, op):
    if not _is_float(x.dtype):
        raise TypeError(f"{op} needs a floating-point tensor, got {x.dtype}: convert it with .to(dtype)")
    return x


def _common_dtype(op, operands):
    """The dtype the tensors among ``operands`` share (weakly typed ones,
    runtime scalars, take the others')."""
    tensors = [x for x in map(_lift, operands) if isinstance(x, TracedTensor)]
    dtypes = sorted({x.dtype for x in tensors if not x.weak} or {x.dtype for x in tensors})
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
    x = _lift(x)
    if isinstance(x, TracedTensor) and x.weak and x.dtype != dtype:
        # A runtime scalar (float32): converted, as a Python float would be.
        if not _is_float(dtype):
            raise TypeError(f"{op} of a {dtype} tensor and a float: convert the tensor with .to(dtype)")
        return x.to(dtype)
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
    """``prim`` on ``operands``, of one dtype, broadcast to a common shape:
    weakly typed if every tensor among them is (``eps * 2``)."""
    out = prim(*_operands(prim.__name__.lstrip("_"), *operands))
    tensors = [x for x in operands if isinstance(x, TracedTensor)]
    out.weak = bool(tensors) and all(x.weak for x in tensors)
    return out


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
        other = _lift(other)
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

    __slots__ = ("graph", "var", "dtype", "shape", "weak")

    def __init__(self, graph, var):
        self.graph = graph
        self.var = var
        # A runtime scalar's (a float argument's): it takes its tensor
        # operand's dtype, as a Python scalar does.
        self.weak = False
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
        other = _lift(other)
        return matmul(self, other) if isinstance(other, TracedTensor) else NotImplemented

    def __rmatmul__(self, other):
        other = _lift(other)
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

    def sqrt(self):
        return prims.sqrt(_require_float(self, "sqrt"))

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
        """Sum over ``dim`` (all dimensions if None) of the tensor (or it
        converted to ``dtype``), accumulated in float32 for floats (their
        own dtype if wider), the result's dtype; integers in theirs (they
        wrap)."""
        x = self.to(dtype) if dtype else self
        dims = _dims(dim, self.ndim)
        return self._keep(prims.reduce_sum(x, dims, _accum_dtype(x.dtype)), dims, keepdim)

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
        other = _lift(other)
        if isinstance(other, TracedTensor):
            return maximum(self, other)
        raise NotImplementedError("max(dim) returns indices, which lumen does not support yet; use amax(dim)")

    def softmax(self, dim, dtype=None):
        x = _require_float(self.to(dtype) if dtype else self, "softmax")
        if _dim(dim, x.ndim) == x.ndim - 1:
            # One primitive: one kernel (online softmax) on MPS.
            return prims.softmax(x, x.ndim - 1)
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

    # -- indexing and splitting ----------------------------------------------

    def _slice(self, starts, limits):
        if list(starts) == [0] * self.ndim and list(limits) == list(self.shape):
            return self
        return prims.slice(self, starts, limits)

    def __getitem__(self, key):
        """Basic indexing (torch, NumPy): integers (which drop their
        dimension), slices with unit steps, and one ``...``."""
        key = key if isinstance(key, tuple) else (key,)
        ellipses = [i for i, k in enumerate(key) if k is Ellipsis]
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(key) - len(ellipses) > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")
        at = ellipses[0] if ellipses else len(key)
        fill = (slice(None),) * (self.ndim - (len(key) - len(ellipses)))
        key = key[:at] + fill + key[at + len(ellipses):]
        starts, limits, shape = [], [], []
        for d, (k, n) in enumerate(zip(key, self.shape)):
            if isinstance(k, slice):
                start, stop, step = k.indices(n)
                if step != 1:
                    raise NotImplementedError("slices with steps other than 1 are not supported")
                starts.append(start)
                limits.append(builtins.max(start, stop))
                shape.append(limits[-1] - start)
            elif isinstance(k, int) and not isinstance(k, bool):
                if not -n <= k < n:
                    raise IndexError(f"index {k} is out of bounds for dimension {d} with size {n}")
                starts.append(k % n)
                limits.append(k % n + 1)
            else:
                raise TypeError(f"indices must be integers, slices or '...', got {type(k).__name__}")
        out = self._slice(starts, limits)
        return out if tuple(shape) == out.shape else out.reshape(shape)

    def narrow(self, dim, start, length):
        """The ``length`` elements of dimension ``dim`` from ``start``."""
        d = _dim(dim, self.ndim)
        n = self.shape[d]
        start = start + n if start < 0 else start
        if not (0 <= start and length >= 0 and start + length <= n):
            raise IndexError(f"narrow: [{start}, {start + length}) is not within dimension {d} of size {n}")
        starts = [0] * self.ndim
        limits = list(self.shape)
        starts[d], limits[d] = start, start + length
        return self._slice(starts, limits)

    def split(self, split_size_or_sections, dim=0):
        """Pieces along ``dim`` (torch): of ``split_size_or_sections``
        elements each (the last may be smaller), or of the given sizes."""
        d = _dim(dim, self.ndim)
        n = self.shape[d]
        if isinstance(split_size_or_sections, int):
            size = split_size_or_sections
            if size <= 0:
                raise RuntimeError(f"split expects split_size be positive, but got split_size={size}")
            sizes = [builtins.min(size, n - start) for start in range(0, n, size)] or [0]
        else:
            sizes = list(split_size_or_sections)
            if sum(sizes) != n:
                raise RuntimeError(
                    f"split_with_sizes expects split_sizes to sum exactly to {n} (input tensor's size at "
                    f"dimension {d}), but got split_sizes={sizes}"
                )
        pieces, start = [], 0
        for size in sizes:
            pieces.append(self.narrow(d, start, size))
            start += size
        return tuple(pieces)

    def chunk(self, chunks, dim=0):
        """Up to ``chunks`` equal pieces along ``dim`` (torch: each of
        ``ceil(size / chunks)`` elements, the last possibly smaller)."""
        if chunks <= 0:
            raise RuntimeError(f"chunk expects `chunks` to be greater than 0, got: {chunks}")
        n = self.shape[_dim(dim, self.ndim)]
        return self.split(builtins.max(-(-n // chunks), 1), dim)


# ---------------------------------------------------------------------
# functions (torch.where, torch.matmul, ...)
# ---------------------------------------------------------------------


def where(condition, input, other):
    condition, input, other = _lift(condition), _lift(input), _lift(other)
    if not isinstance(condition, TracedTensor) or condition.dtype != "bool":
        raise TypeError("where expected condition to be a bool tensor")
    dtype = _common_dtype("where", (input, other))
    input, other = _as_tensor(input, dtype, "where"), _as_tensor(other, dtype, "where")
    shape = _broadcast_shapes(condition.shape, input.shape, other.shape)
    return prims.select(*(_broadcast_to(t, shape) for t in (condition, input, other)))


def matmul(input, other):
    """``input @ other`` with torch's rules: 1-d operands are vectors, and
    the dimensions before the last two are batch dimensions, broadcast. It
    accumulates floats in float32 (float64 in float64); its result is of
    the inputs' dtype."""
    input, other = _lift(input), _lift(other)
    _common_dtype("matmul", (input, other))
    if input.ndim == 0 or other.ndim == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D")
    x = input.unsqueeze(0) if input.ndim == 1 else input
    y = other.unsqueeze(-1) if other.ndim == 1 else other
    batch = _broadcast_shapes(x.shape[:-2], y.shape[:-2])
    x, y = _broadcast_to(x, batch + x.shape[-2:]), _broadcast_to(y, batch + y.shape[-2:])
    b = tuple(range(len(batch)))
    # Accumulated in float32 (or wider), the result in the inputs' dtype.
    dims = (((len(b) + 1,), (len(b),)), (b, b))
    out = prims.dot_general(x, y, dims, _accum_dtype(x.dtype), x.dtype)
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


def sqrt(input):
    return input.sqrt()



def tanh(input):
    return input.tanh()


def sigmoid(input):
    return input.sigmoid()


def softmax(input, dim, dtype=None):
    return input.softmax(dim, dtype)


def rms_norm(input, normalized_shape, weight=None, eps=None):
    """``torch.nn.functional.rms_norm``: ``input`` normalized by its root mean
    square over the last dimension (``normalized_shape``, its size),
    ``input / sqrt(mean(input^2) + eps)``, times ``weight`` if given. ``eps``
    defaults to the dtype's machine epsilon, as in torch. Traced as its
    primitives, which the MPS compiler recognizes (as it does an RMS norm
    written by hand) and runs as one kernel, with the ops computing
    ``input`` fused in."""

    x = _require_float(_lift(input), "rms_norm")
    shape = [normalized_shape] if isinstance(normalized_shape, int) else list(normalized_shape)

    if shape != list(x.shape[-1:]):
        raise NotImplementedError(f"rms_norm normalizes the last dimension, of size {x.shape[-1:]}, got {shape}")

    weight = _lift(weight)
    if weight is not None and (weight.dtype != x.dtype or list(weight.shape) != shape):
        raise TypeError(f"rms_norm: weight must be {x.dtype}{shape}, got {weight.dtype}{list(weight.shape)}")

    if eps is None:
        eps = {"float16": 2.0**-10, "bfloat16": 2.0**-7, "float64": 2.0**-52}.get(x.dtype, 2.0**-23)

    # Normalized in the accumulation dtype (float32 for narrower floats, as
    # its mean is), then cast back and scaled by the weight in x's dtype.
    h = x.to(_accum_dtype(x.dtype))
    y = (h / ((h * h).mean(-1, keepdim=True) + eps).sqrt()).to(x.dtype)

    if weight is not None:
        y = y * weight

    return y
