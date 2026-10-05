"""Tracing, and traced tensors.

``lumen.compile(fn)`` runs ``fn`` once per input signature with a
``TracedTensor`` standing in for each input tensor, recording the ops it
applies into a graph (JAX: a jaxpr), which then runs as a whole. Only
traced tensors have ops: a ``lumen.Tensor`` holds data, and the graph is
the only thing that computes with it.

A traced tensor has PyTorch's operators (``+ - * / @ == < []``, ``-x``),
layout methods (``reshape``, ``permute``, ``transpose``, ...) and dtype
casts (``to``, ``float``, ...); every other op is a function in
``lumen.functional`` (``import lumen.functional as F``). Each is a
composition of the strict primitives in ``lumen.graph.prims``. Unlike
PyTorch, no op changes a dtype the user did not ask for: operands must
share a dtype (a Python scalar takes its tensor operand's, and must be of
its kind), and floating-point functions take floating-point tensors.
Convert with ``.to(dtype)``.
"""

import builtins
import contextlib
import dataclasses
import functools
import math
import os
import sys
import warnings

from lumen import nn
from lumen._C import Graph, Plan, Tensor, _pack, config
from lumen.graph import prims

__all__ = [
    "TracedTensor",
    "compile",
    "make_graph",
]

# The graphs being traced, innermost last.
_TRACES = []
# Each trace's random state (``lumen.random``): its hidden inputs, the seed
# and the stream's position, made at its first draw (None before), and the
# numbers drawn so far.
_RNG = []
# Each trace's device: where its values are, unless on the host (``.cpu()``).
_DEVICES = []
# Each trace's open ``record_function`` ranges (``lumen.profiler``),
# outermost first: the scope of the nodes traced inside them, whose steps
# the profiler shows inside them when the plan runs.
_SCOPES = []
# Each trace's values' scopes (a tuple of range names), those traced inside
# a range: their gradient ops' are each name's backward (``_backward_scope``).
_SCOPE_OF = []
# Each trace's values' source lines (``(filename, lineno)``): the line
# outside lumen that computed each.
_SOURCES = []


# Frames in lumen's package are lumen's, not the traced program's.
_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The autodiff tapes being recorded (``lumen/autograd``), innermost
# last: each node bound while one is open, ``(name, input vars, params,
# output var)``, is appended to each.
_TAPES = []
# Each trace's whole tape and the tensors ``.backward()`` differentiates
# with respect to (its tensor arguments and weights), innermost last.
_BACKWARD = []


def _record(name, inputs, params, var):
    """Record node ``var = name(inputs, **params)``: its source line, and
    on each open tape."""
    for tape in _TAPES:
        tape.append((name, inputs, params, var))
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back
    if frame is not None:
        _SOURCES[-1][var] = (frame.f_code.co_filename, frame.f_lineno)
    if _SCOPES and _SCOPES[-1]:
        _SCOPE_OF[-1][var] = tuple(_SCOPES[-1])


def _is_device(x):
    """Whether ``x`` names a device (``"cpu"``, ``"mps"``, ``"cuda:0"``, a
    ``lumen.device``), not a dtype."""
    from lumen._C import device

    return isinstance(x, device) or (isinstance(x, str) and x.split(":")[0] in ("cpu", "mps", "cuda", "meta"))


def _tree_map(f, x):
    """``x`` with each traced tensor in it (through modules, lists and
    tuples) replaced by ``f(t)``."""
    if isinstance(x, TracedTensor):
        return f(x)
    if isinstance(x, nn.Module):
        fields = {fl.name: _tree_map(f, getattr(x, fl.name)) for fl in dataclasses.fields(x)}
        return dataclasses.replace(x, **fields)
    if isinstance(x, (list, tuple)):
        return type(x)(_tree_map(f, v) for v in x)
    return x


def _tree_leaves(x):
    """The traced tensors in ``x``, in ``_tree_map``'s order."""
    leaves = []
    _tree_map(leaves.append, x)
    return leaves


def _random_bits(shape):
    """uint32 random bits of ``shape`` (``lumen.random``): the next run of
    the stream, from the trace's random state: two hidden inputs of the
    graph, after the weights, made at the first draw (the seed and the
    stream's position, uint64 runtime scalars: kernels take them by value)."""
    graph = current_graph()
    rng = _RNG[-1]
    if rng["state"] is None:
        rng["state"] = [TracedTensor(graph, graph.input("uint64", [])) for _ in range(2)]
    offset = rng["drawn"]
    rng["drawn"] += math.prod(shape)
    return prims.random_bits(*rng["state"], shape, offset)


def _enter_scope(name):
    """Open ``record_function`` range ``name`` in the trace, if one is
    running: the nodes traced until it closes are in it."""
    if _TRACES:
        _SCOPES[-1].append(name)
        _TRACES[-1]._set_scope(_SCOPES[-1])


def _exit_scope():
    """Close the trace's innermost ``record_function`` range, if one runs."""
    if _TRACES:
        _SCOPES[-1].pop()
        _TRACES[-1]._set_scope(_SCOPES[-1])


def _backward_scope(var, base):
    """The scope of value ``var``'s gradient ops: ``base`` (the ranges open
    where the backward runs), then each range ``var`` was traced in as
    ``name (backward)``."""
    return tuple(base) + tuple(f"{name} (backward)" for name in _SCOPE_OF[-1].get(var, ()))


@contextlib.contextmanager
def _scope(names):
    """Trace inside the ``record_function`` ranges ``names`` (outermost
    first), the open ones back after."""
    saved = list(_SCOPES[-1])
    _SCOPES[-1][:] = names
    _TRACES[-1]._set_scope(_SCOPES[-1])
    try:
        yield
    finally:
        _SCOPES[-1][:] = saved
        _TRACES[-1]._set_scope(saved)


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
                    raise TypeError(
                        f"a module's weights must be whole meta tensors, got a {t.device} tensor of shape {t.shape}"
                    )
                params.setdefault(t.storage_id, t)
    return list(params.values())


def _scalars(args):
    """The runtime scalars of ``args``: the float arguments, then the float
    fields of the module arguments, in order."""
    floats = [a for a in args if isinstance(a, float)]
    return floats + [v for a in args if isinstance(a, nn.Module) for v in nn._floats(a)]


def _trace(fn, args, device):
    """``fn`` traced on ``args``: the graph, and what ``fn`` returned (a
    traced tensor, or modules, lists and tuples of them: the structure the
    graph's outputs, its traced tensors in order, are returned in). The graph's inputs
    are the tensor arguments, then the runtime scalars (``_scalars``: 0-d
    float32 inputs, weakly typed: each takes its tensor operand's dtype),
    then the weights of the module arguments (``_weights``), each a whole
    meta tensor, then, if ``fn`` draws random numbers (``lumen.random``),
    the random state (the seed and the stream's position, uint64 runtime
    scalars); the numbers it draws are
    returned last (None if none). Precision warnings point at the line computing the value
    they are about."""
    graph = Graph()
    traced = [TracedTensor(graph, graph.input(a.dtype, a.shape), device) if isinstance(a, Tensor) else a for a in args]

    def scalar(_):
        t = TracedTensor(graph, graph.input("float32", []), device)
        t.weak = True
        return t

    traced = [scalar(a) if isinstance(a, float) else a for a in traced]
    scalars = [scalar(v) for a in args if isinstance(a, nn.Module) for v in nn._floats(a)]
    weights = {}
    for t in _weights(args):
        weights[t.storage_id] = TracedTensor(graph, graph.input(t.dtype, t.shape), device)
    module_scalars = iter(scalars)
    traced = [
        (
            nn.map_tensors(a, lambda t: weights[t.storage_id], lambda _: next(module_scalars))
            if isinstance(a, nn.Module)
            else a
        )
        for a in traced
    ]

    sources = {}
    arguments = [t for t in traced if isinstance(t, TracedTensor) and not t.weak]
    assigned = {sid: t.var for sid, t in weights.items()}
    assigned_scalars = [t.var for t in scalars]
    assigned_arguments = [t.var for t in arguments]
    # Recorded from the start, for ``.backward()``.
    tape = []
    leaves = [t for t in traced if isinstance(t, TracedTensor) and not t.weak] + list(weights.values())

    _TRACES.append(graph)
    _SCOPES.append([])
    _SCOPE_OF.append({})
    _DEVICES.append(device)
    _RNG.append({"state": None, "drawn": 0})
    _SOURCES.append(sources)
    _TAPES.append(tape)
    _BACKWARD.append((tape, leaves))

    try:
        out = fn(*traced)
    finally:
        _TRACES.pop()
        _SCOPES.pop()
        _SCOPE_OF.pop()
        _DEVICES.pop()
        rng = _RNG.pop()
        _SOURCES.pop()
        _TAPES.pop()
        _BACKWARD.pop()

    outputs = _tree_leaves(out)
    if not isinstance(out, (TracedTensor, nn.Module, list, tuple)) or any(o.graph is not graph for o in outputs):
        raise TypeError(f"a compiled function must return tensors computed from its inputs, got {out!r}")
    # A weight, a module's float or a tensor argument ``fn`` assigned
    # (``w.copy_(value)``, or through a view of it): its new value is an
    # output too, written back after each run (the float on the host).
    written = [k for k, (sid, t) in enumerate(weights.items()) if t.var != assigned[sid]]
    written_scalars = [k for k, t in enumerate(scalars) if t.var != assigned_scalars[k]]
    written_arguments = [k for k, t in enumerate(arguments) if t.var != assigned_arguments[k]]
    graph.set_outputs(
        [o.var for o in outputs]
        + [list(weights.values())[k].var for k in written]
        + [scalars[k].var for k in written_scalars]
        + [arguments[k].var for k in written_arguments]
    )
    # Without the values no output depends on (an autodiff transform's
    # tangents its transpose does not read).
    renumbered = graph.prune()
    sources = {renumbered[v]: s for v, s in sources.items() if renumbered[v] is not None}
    grads = [t.grad for t in leaves if t.grad is not None]
    if grads and all(renumbered[g.var] is None for g in grads):
        warnings.warn(
            "backward() computed gradients the compiled function does not use: it is pruned. "
            "Return them (x.grad, [p.grad for p in model.parameters()]) or use them (opt.step())",
            stacklevel=3,
        )
    for var, cast, message in graph.precision_warnings():
        where, cast_at = sources.get(var), sources.get(cast)
        if cast_at and cast_at != where:
            message += f" (cast at {cast_at[0]}:{cast_at[1]})"
        if where:
            warnings.warn_explicit(message, UserWarning, *where)
        else:
            warnings.warn(message)
    drawn = rng["drawn"] if rng["state"] is not None else None
    return graph, out, (written, written_scalars, written_arguments), drawn


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
        (
            (a.dtype, tuple(a.shape))
            if isinstance(a, Tensor)
            else nn.structure(a) if isinstance(a, (nn.Module, float)) else ("static", a)
        )
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

    A weight the function assigns (``w.copy_(value)``, an optimizer's
    step) is written back into the weight's memory after each call; a
    module's float so assigned (a step count), into the module.

    ``compiled.dump_graph(path)`` writes the graph and plan of the latest
    call's signature, profiled, as an HTML page (``lumen.graph.viz``);
    ``compiled.dump_graph(path, *args)``, those of ``args``' signature.

    With ``lumen.config.compiler.neural_engine``, an MPS plan runs the large
    float16 dots of the weights the function does not write (and the
    float16 work after them) on the Apple Neural Engine, their values baked
    into Core ML programs: compiled again once a weight is written
    (``copy_``, or by another compiled function)."""
    compile_device = device  # dump_graph's own `device` shadows it
    # Per signature and device: the graph, whether `fn` returns one tensor,
    # its plans, each with its workspace (`None` on the meta device), tried
    # in order, and the number of tensor arguments (the graph's inputs
    # before the weights).
    plans = {}
    latest = []

    def prepare(args):
        """The plan for ``args`` and its inputs: the graph, the plan, its
        workspace, what ``fn`` returns (its outputs' structure), and the plan's
        inputs (the arguments, then the weights placed, and their blocks)."""
        target = _device(args, device)
        # The compiler's flags too: a plan compiled with others is not reused.
        key = _signature(args), target, repr(config.compiler)
        # The arguments the plan copies in: tensors, then runtime scalars.
        # The kernels take the scalars by value, read from the host on each
        # call: a new value needs no new plan.
        tensors = [a for a in args if isinstance(a, Tensor)]
        scalars = list(range(len(tensors), len(tensors) + len(_scalars(args))))
        tensors += [Tensor.full([], v, "float32") for v in _scalars(args)]
        if key not in plans:
            plans[key] = (*_trace(fn, args, target), [], len(tensors), scalars)
        graph, out, _, drawn, entries, _, _ = plans[key]
        # The random state, if it draws numbers: inputs after the weights,
        # runtime scalars.
        from lumen import random  # lazily: lumen.random imports this module

        rng = random._state() if drawn is not None else []
        weights = _weights(args)
        latest[:] = [key]
        if target == "meta":
            if not entries:
                entries.append((Plan(graph, "meta"), None))
            return graph, entries[0][0], None, out, tensors + weights + rng
        params = list(range(len(tensors), len(tensors) + len(weights)))
        # The random state's scalars, after the weights.
        start = len(tensors) + len(weights)
        scalars = scalars + list(range(start, start + len(rng)))
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
            return tensors + [w._placed(target) for w in weights] + rng + blocks

        for plan, workspace in entries:
            placed = inputs_for(plan)
            if placed is not None:
                return graph, plan, workspace, out, placed
        # A new plan: dots merged into blocks of weights while none of theirs
        # is placed yet; once they are placed otherwise, without.
        packable = params if not entries else []
        # The weights it assigns, each donated to its new value (an output
        # after those it returns): written over the weight where nothing
        # reads the old one after (in place: a cache's dynamic_update_slice),
        # not copied back; no other output is written there.
        written, written_scalars, written_arguments = plans[key][2]
        returned = len(graph.outputs()) - len(written) - len(written_scalars) - len(written_arguments)
        donate = [(len(tensors) + w, returned + j) for j, w in enumerate(written)]
        plan = Plan(graph, target, donate_into=donate, parameters=params, packable=packable, scalars=scalars)
        workspace = Tensor.empty([plan.workspace_bytes], "uint8", target)
        entries.append((plan, workspace))
        placed = inputs_for(plan)
        if placed is None:
            plan = Plan(graph, target, donate_into=donate, parameters=params, scalars=scalars)
            entries[-1] = plan, workspace
            placed = inputs_for(plan)
        return graph, plan, workspace, out, placed

    # Per plan running Neural Engine steps: its weights' versions when it
    # last ran (their values baked into its programs).
    baked = {}

    @functools.wraps(fn)
    def compiled(*args):
        graph, plan, workspace, out, inputs = prepare(args)
        ran = not (workspace is None or any(isinstance(a, Tensor) and a.device == "meta" for a in args))
        if ran and plan.neural_engine:
            versions = [w._version for w in _weights(args)]
            if baked.get(id(plan)) != versions:
                plan._invalidate_neural_engine()
                baked[id(plan)] = versions
        if not ran:
            # Compiled (and the weights placed); nothing to run.
            outputs = [
                Tensor.empty(list(shape), dtype, "meta") for dtype, shape in map(graph.type_of, graph.outputs())
            ]
        else:
            outputs = plan.run_in(workspace, inputs)
        outputs = iter(outputs)

        # A host value the plan has on the device (an input or a constant as
        # it is: no host op computed it), copied.
        def result_of(t):
            value = next(outputs)
            return value.cpu() if ran and t.device == "cpu" and value.device != "cpu" else value

        result = _tree_map(result_of, out)
        if ran:
            # The weights it assigned, written back (functionalized: as
            # torch.compile does a mutation).
            weights = _weights(args)
            written, written_scalars, written_arguments = plans[latest[0]][2]
            target = latest[0][1]
            for k, value in zip(written, outputs):
                # Written over its memory already (donated), or copied.
                w = weights[k]
                memory = w._placed(target) if w._is_parameter else w
                if value.storage_id != memory.storage_id:
                    w.copy_(value)
                elif w._is_parameter:
                    w._mark_written()
            for k, value in zip(written_scalars, outputs):
                (v,) = value.tolist()  # a 0-d tensor lists its one element
                nn._set_float(args, k, float(v))
            arguments = [a for a in args if isinstance(a, Tensor)]
            for k, value in zip(written_arguments, outputs):
                arguments[k].copy_(value)
            # The next call draws the numbers after this one's.
            drawn = plans[latest[0]][3]
            if drawn is not None:
                from lumen import random

                random._advance(drawn)
        return result

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
        ((_, target, _),) = latest
        graph, _, _, drawn, _, n_tensors, scalars = plans[latest[0]]
        target = str(device or compile_device or (target if target != "meta" else "cpu"))
        # Kernels do not depend on the values: profile on ones, the weights
        # packed where the compiler merges dots.
        types = [graph.type_of(v) for v in graph.inputs()]
        # The weights: the inputs after the tensors but the random state,
        # whose two scalars are runtime scalars.
        n_rng = 2 if drawn is not None else 0
        params = list(range(n_tensors, len(types) - n_rng))
        scalars = scalars + list(range(len(types) - n_rng, len(types)))
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
            graph,
            plan,
            inputs,
            title=fn.__name__,
            runs=runs,
            device=target,
            run=lambda: plan.run_in(workspace, inputs),
            scalars=scalars,
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
        return _trace(fn, args, _device(args, None))[0]

    return graph


# ---------------------------------------------------------------------
# dtypes: never changed implicitly
# ---------------------------------------------------------------------


def _is_float(dtype):
    return "float" in dtype


def _accum_dtype(dtype):
    """The dtype a matmul of ``dtype`` accumulates in: float32 for floats,
    or their own if wider (float64); integers their own."""
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
        raise TypeError(
            f"{op} of a {dtype} tensor and the {type(x).__name__} {x!r}: convert the tensor with .to(dtype)"
        )
    return prims.full((), x, dtype)


def _broadcast_shapes(*shapes):
    ndim = builtins.max(len(s) for s in shapes)
    out = []
    for d in range(-ndim, 0):
        sizes = {s[d] for s in shapes if len(s) >= -d} - {1}
        if len(sizes) > 1:
            raise RuntimeError(
                f"shapes {', '.join(map(str, map(list, shapes)))} are not broadcastable at dimension {d}"
            )
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
    # By a scalar (a Python number, a runtime scalar): times its reciprocal.
    if isinstance(y, (int, float)) and not isinstance(y, bool) and y != 0:
        return _elementwise(prims.mul, _require_float(x, "true division (/)"), 1 / y)
    if isinstance(y, TracedTensor) and y.weak and not getattr(x, "weak", True):
        r = _true_div(1.0, y)
        r.weak = True
        return _elementwise(prims.mul, _require_float(x, "true division (/)"), r)
    x, y = _operands("div", x, y)
    return prims.div(_require_float(x, "true division (/)"), y)


def _view(x, out, op):
    """``out``, ``op`` (a reshape, permute or slice) of ``x``, as a view of
    ``x``'s base: it reads the base's assignments (``copy_``), and its own
    reach the base."""
    base = x._base if x._base is not None else x
    out._base, out._ops, out._seen = base, x._ops + (op,), base._version
    return out


def _strides(x):
    """``x``'s strides as torch's view of its base (contiguous) would have
    them: its ops' strides from the base's."""
    base = x._base if x._base is not None else x
    shape = list(base.shape)
    strides = [math.prod(shape[d + 1 :]) for d in range(len(shape))]
    for op in x._ops:
        if op[0] == "reshape":
            strides, shape = _reshape_strides(shape, strides, op[1]), list(op[1])
        elif op[0] == "permute":
            strides, shape = [strides[d] for d in op[1]], [shape[d] for d in op[1]]
        else:
            shape = [limit - start for start, limit in zip(op[1], op[2])]
    return strides


def _reshape_strides(shape, strides, new_shape):
    """The strides of a view of ``new_shape`` of a tensor of ``shape`` at
    ``strides``, or None if there is none (torch: ``computeStride``)."""
    if not shape:
        return [1] * len(new_shape)
    if math.prod(shape) == 0:
        return [math.prod(new_shape[d + 1 :]) for d in range(len(new_shape))]
    new_strides = [0] * len(new_shape)
    view_d = len(new_shape) - 1
    # The stride of the last dimension of the chunk being matched (a run
    # of contiguous dimensions), and the elements matched on each side.
    chunk_stride, tensor_numel, view_numel = strides[-1], 1, 1
    for d in range(len(shape) - 1, -1, -1):
        tensor_numel *= shape[d]
        if d == 0 or (shape[d - 1] != 1 and strides[d - 1] != tensor_numel * chunk_stride):
            while view_d >= 0 and (view_numel < tensor_numel or new_shape[view_d] == 1):
                new_strides[view_d] = view_numel * chunk_stride
                view_numel *= new_shape[view_d]
                view_d -= 1
            if view_numel != tensor_numel:
                return None
            if d > 0:
                chunk_stride, tensor_numel, view_numel = strides[d - 1], 1, 1
    return new_strides if view_d == -1 else None


def _apply(x, op):
    """``op`` of ``x``: a view op, as recorded by ``_view``."""
    if op[0] == "reshape":
        return prims.reshape(x, op[1])
    if op[0] == "permute":
        return prims.transpose(x, op[1])
    return prims.slice(x, op[1], op[2])


def _derive(base, ops):
    """The view ``ops`` derive from ``base``'s current value."""
    for op in ops:
        base = _apply(base, op)
    return base


def _scatter(x, ops, value):
    """``x`` with the part ``ops`` derive from it replaced by ``value``: a
    reshape's back, a permute's inverse, a slice's written at its start
    (``dynamic_update_slice``)."""
    if not ops:
        return value
    op = ops[0]
    new = _scatter(_apply(x, op), ops[1:], value)
    if op[0] == "reshape":
        return prims.reshape(new, x.shape)
    if op[0] == "permute":
        return prims.transpose(new, sorted(range(len(op[1])), key=op[1].__getitem__))
    return prims.dynamic_update_slice(x, new, op[1])


class TracedTensor:
    """A value of the graph being traced, standing in for a tensor inside a
    function passed to ``lumen.compile``. It has a dtype and shape but no
    data; its methods record primitives into the graph."""

    __slots__ = ("graph", "_var", "dtype", "shape", "weak", "grad", "_base", "_ops", "_seen", "_version", "device")

    def __init__(self, graph, var, device=None):
        self.graph = graph
        self._var = var
        # A view's (torch's: ``reshape``, ``permute``, indexing): the tensor
        # it is a view of and the ops deriving it from that, the base's
        # version its value is of; a base's, how often it was assigned
        # (``copy_``, its own or a view's).
        self._base, self._ops, self._seen, self._version = None, (), 0, 0
        # A runtime scalar's (a float argument's): it takes its tensor
        # operand's dtype, as a Python scalar does.
        self.weak = False
        # Its gradient, once ``.backward()`` computed one (torch's ``.grad``).
        self.grad = None
        # Where it is: the trace's device, or the host (``"cpu"``: ``.cpu()``
        # made it, or an op of host values).
        self.device = device or (_DEVICES[-1] if _DEVICES else None)
        dtype, shape = graph.type_of(var)
        self.dtype = dtype
        self.shape = tuple(shape)

    @property
    def var(self):
        """Its value in the graph. A view's is derived again from its base's
        once the base, or another of its views, is assigned (``copy_``),
        as torch's view of a tensor modified in place reads the change."""
        base = self._base
        if base is not None and self._seen != base._version:
            self._var, self._seen = _derive(base, self._ops).var, base._version
        return self._var

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

    def copy_(self, value):
        """Assign ``value`` (of its type) to it, in place (torch's ``copy_``):
        every reference to it reads ``value`` from here on, and so does
        every view of the same tensor (a view's part of its base). A
        module's weight or a tensor argument so assigned is written back
        into it after each call of the compiled function (an optimizer's
        step)."""
        value = _lift(value)
        if not isinstance(value, TracedTensor) or (value.dtype, value.shape) != (self.dtype, self.shape):
            raise TypeError(f"copy_: needs a tensor of its type, {self.dtype}{list(self.shape)}, got {value!r}")
        # Functionalized (torch.compile's): a view's assignment is its
        # base's, the view's part of it replaced (each view of the base
        # derived again when read).
        base = self._base
        if base is None:
            self._var = value.var
            self._version += 1
        else:
            base._var = _scatter(base, self._ops, value).var
            base._version += 1
            self._var, self._seen = value.var, base._version
        return self

    # -- in place (functionalized: each an out-of-place op, then ``copy_``) --

    def _assign(self, value, op):
        """Assign ``value``, ``op`` of it computed out of place, as torch's
        in-place ``op``: of its shape (the other operand broadcast to it,
        never it to the other's) and dtype."""
        if value.shape != self.shape:
            raise RuntimeError(
                f"{op}: output with shape {list(self.shape)} doesn't match the broadcast shape {list(value.shape)}"
            )
        if value.dtype != self.dtype:
            raise TypeError(f"{op}: result type {value.dtype} can't be cast to the desired output type {self.dtype}")
        return self.copy_(value)

    def add_(self, other, alpha=1):
        return self._assign(self + (other if alpha == 1 else other * alpha), "add_")

    def sub_(self, other, alpha=1):
        return self._assign(self - (other if alpha == 1 else other * alpha), "sub_")

    def mul_(self, other):
        return self._assign(self * other, "mul_")

    def div_(self, other):
        return self._assign(self / other, "div_")

    def neg_(self):
        return self._assign(-self, "neg_")

    def exp_(self):
        return self._assign(prims.exp(_require_float(self, "exp_")), "exp_")

    def log_(self):
        return self._assign(prims.log(_require_float(self, "log_")), "log_")

    def sqrt_(self):
        return self._assign(prims.sqrt(_require_float(self, "sqrt_")), "sqrt_")

    def tanh_(self):
        return self._assign(prims.tanh(_require_float(self, "tanh_")), "tanh_")

    def sigmoid_(self):
        return self._assign(prims.logistic(_require_float(self, "sigmoid_")), "sigmoid_")

    def relu_(self):
        return self._assign(_elementwise(prims.max, self, 0), "relu_")

    def clamp_(self, min=None, max=None):
        if min is None and max is None:
            raise RuntimeError("clamp_: at least one of 'min' or 'max' must not be None")
        value = self if min is None else _elementwise(prims.max, self, min)
        value = value if max is None else _elementwise(_min, value, max)
        return self._assign(value, "clamp_")

    def clamp_min_(self, min):
        return self.clamp_(min=min)

    def clamp_max_(self, max):
        return self.clamp_(max=max)

    def uniform_(self, from_=0.0, to=1.0):
        """Fill it with uniform random numbers in ``[from_, to)``
        (``Tensor.uniform_``, its ``from`` here ``from_``), drawn from the
        generator (``lumen.random``): a weight so filled inside a compiled
        function is written back after the call (an initialization)."""
        from lumen import random

        return self._assign(random._uniform(self.shape, self.dtype, from_, to), "uniform_")

    def zero_(self):
        return self.fill_(False if self.dtype == "bool" else 0)

    def fill_(self, value):
        value = _as_tensor(value, self.dtype, "fill_")
        if value.shape != ():
            raise RuntimeError(
                f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
            )
        return self._assign(_broadcast_to(value, self.shape), "fill_")

    def masked_fill_(self, mask, value):
        """``value`` where ``mask`` (a bool tensor broadcast to it)."""
        mask = _lift(mask)
        if not isinstance(mask, TracedTensor) or mask.dtype != "bool":
            raise TypeError("masked_fill_ only supports boolean masks")
        value = _broadcast_to(_as_tensor(value, self.dtype, "masked_fill_"), self.shape)
        mask = _broadcast_to(mask, _broadcast_shapes(mask.shape, self.shape))
        return self._assign(prims.select(mask, value, self), "masked_fill_")

    # ``x += y``: in place (torch's), not ``x = x + y``.
    def __iadd__(self, other):
        return self.add_(other)

    def __isub__(self, other):
        return self.sub_(other)

    def __imul__(self, other):
        return self.mul_(other)

    def __itruediv__(self, other):
        return self.div_(other)

    def __setitem__(self, key, value):
        """``x[key] = value``: the indexed view of it assigned ``value`` (a
        Python scalar, or a tensor of its dtype broadcast to the view)."""
        view = self[key]
        # ``y[k] += v``: Python's ``t = y[k]; t += v; y[k] = t``, t already
        # written through (the same part of the same base, up to date).
        base = view._base
        if isinstance(value, TracedTensor) and value is not view and base is not None:
            if value._base is base and value._ops == view._ops and value._seen == base._version:
                return
        value = _as_tensor(value, self.dtype, "index assignment")
        if value.dtype != self.dtype:
            raise TypeError(f"index assignment: got a {value.dtype} value for a {self.dtype} tensor")
        # As torch: the value's leading size-1 dimensions dropped, then it
        # broadcast to the view.
        shape = value.shape
        while len(shape) > view.ndim and shape[0] == 1:
            shape = shape[1:]
        try:
            fits = len(shape) <= view.ndim and _broadcast_shapes(shape, view.shape) == view.shape
        except RuntimeError:
            fits = False
        if not fits:
            raise RuntimeError(
                f"shape mismatch: value tensor of shape {list(value.shape)} cannot be broadcast to indexing result of shape {list(view.shape)}"
            )
        view.copy_(_broadcast_to(value.reshape(shape), view.shape))

    def backward(self, gradient=None):
        """Its gradient (``gradient``: its cotangent; for a scalar, 1 by
        default) with respect to the traced function's tensor arguments and
        its modules' weights, added to their ``.grad`` (``torch.Tensor.backward``,
        within the trace: return them). As ``lumen.grad`` computes it."""
        from lumen.autograd.transforms import backward  # it imports this module

        tape, leaves = _BACKWARD[-1]

        if gradient is None:
            if self.shape != () or not _is_float(self.dtype):
                raise RuntimeError(f"backward: a gradient is needed for a non-scalar output, got {self!r}")

            gradient = prims.full((), 1.0, self.dtype)

        for leaf, g in zip(leaves, backward(self, gradient, list(tape), leaves)):
            if g is not None:
                # gradient accumulation
                if leaf.grad is not None:
                    g = leaf.grad + g

                leaf.grad = g

    def __matmul__(self, other):
        """F.matmul, accumulating floats in float32 (float64 in float64),
        its result of the operands' dtype."""
        from lumen.functional import matmul  # it imports this module

        other = _lift(other)
        if not isinstance(other, TracedTensor):
            return NotImplemented
        dtype = _common_dtype("matmul", (self, other))
        return matmul(self, other, _accum_dtype(dtype), dtype)

    def __rmatmul__(self, other):
        other = _lift(other)
        return TracedTensor.__matmul__(other, self) if isinstance(other, TracedTensor) else NotImplemented

    # -- dtype conversion --------------------------------------------------

    def to(self, *args, dtype=None, device=None, non_blocking=False, copy=False):
        """It converted to ``dtype`` and/or on ``device`` (``Tensor.to``: a
        dtype, a device, both, or another tensor, whose dtype it takes): on
        the CPU, :meth:`cpu`; on another, a host value copied to the plan's
        device (``prims.to_device``), a device value as it is."""
        for a in args:
            if isinstance(a, TracedTensor):
                dtype = a.dtype
            elif _is_device(a):
                device = a
            else:
                dtype = a
        out = self if dtype in (None, self.dtype) else prims.cast(self, dtype)
        kind = None if device is None else str(device).split(":")[0]
        if kind == "cpu":
            out = out.cpu()
        elif kind not in (None, "meta") and out.device == "cpu":
            out = prims.to_device(out)
        return out

    def cpu(self):
        """It on the host (``Tensor.cpu``): itself if it is there, else copied
        there (``prims.to_host``: the plan waits for it and copies it). The
        ops reading it run on the host, as PyTorch runs ops on CPU tensors on
        the CPU (reading it and a device value in one op raises); returned,
        it is a CPU tensor."""
        if self.device == "cpu":
            return self
        out = prims.to_host(self)
        out.weak = self.weak
        return out

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
        if tuple(shape) == self.shape:
            return self
        out = prims.reshape(self, shape)
        # As torch's: a view where its strides allow (adding or dropping
        # size-1 dimensions, merging contiguous ones), else a copy (of a
        # permuted tensor, say).
        if _reshape_strides(self.shape, _strides(self), shape) is None:
            return out
        return _view(self, out, ("reshape", tuple(shape)))

    view = reshape

    def flatten(self, start_dim=0, end_dim=-1):
        if self.ndim == 0:
            return self.reshape(1)
        start, end = _dim(start_dim, self.ndim), _dim(end_dim, self.ndim)
        return self.reshape(self.shape[:start] + (-1,) + self.shape[end + 1 :])

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
        return self if dims == sorted(dims) else _view(self, prims.transpose(self, dims), ("permute", tuple(dims)))

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
            raise RuntimeError(
                f"expand: the number of sizes ({len(sizes)}) must be at least the tensor's rank ({self.ndim})"
            )
        shape = tuple(self.shape[i - lead] if n == -1 and i >= lead else n for i, n in enumerate(sizes))
        return _broadcast_to(self, shape)

    def expand_as(self, other):
        return self.expand(other.shape)

    def broadcast_to(self, shape):
        return self.expand(shape)

    # -- indexing and splitting ----------------------------------------------

    def _slice(self, starts, limits):
        if list(starts) == [0] * self.ndim and list(limits) == list(self.shape):
            return self
        return _view(self, prims.slice(self, starts, limits), ("slice", tuple(starts), tuple(limits)))

    def __getitem__(self, key):
        """Basic indexing (torch, NumPy): integers (which drop their
        dimension), slices with unit steps, and one ``...``; or a tensor of
        integer indices into the first dimension (``weight[ids]``, an
        embedding's lookup: ``prims.gather``)."""
        if isinstance(key, TracedTensor):
            return prims.gather(self, key, 0)
        key = key if isinstance(key, tuple) else (key,)
        ellipses = [i for i, k in enumerate(key) if k is Ellipsis]
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        if len(key) - len(ellipses) > self.ndim:
            raise IndexError(f"too many indices for tensor of dimension {self.ndim}")
        at = ellipses[0] if ellipses else len(key)
        fill = (slice(None),) * (self.ndim - (len(key) - len(ellipses)))
        key = key[:at] + fill + key[at + len(ellipses) :]
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
