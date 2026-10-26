"""Tracing Python functions into lumen graphs: ``TracedTensor``, ``compile`` and ``make_graph``."""

import builtins
import contextlib
import dataclasses
import functools
import itertools
import math
import os
import sys
import warnings

from lumen import nn
from lumen.config._C import config
from lumen.graph import prims
from lumen.graph._C import Graph
from lumen.graph_compiler._C import Plan
from lumen.tensor._C import Tensor, _pack

__all__ = [
    "TracedTensor",
    "compile",
    "make_graph",
]

# Stacks with one entry per open ``tracing`` context, innermost last.
_TRACES = []
_RNG = []
_DEVICES = []
_HOSTS = []
_SCOPES = []
_SCOPE_OF = []
_CALLS = itertools.count()
_SOURCES = []
_PRODUCERS = []


_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_TAPES = []
_BACKWARD = []


def _record(name, inputs, params, var):
    """Record node ``var = name(inputs, **params)``: its source line, and
    on each open tape."""
    for tape in _TAPES:
        tape.append((name, inputs, params, var))
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back
    _PRODUCERS[-1][var] = (name, inputs, params)
    if frame is not None:
        _SOURCES[-1][var] = (frame.f_code.co_filename, frame.f_lineno)
        _TRACES[-1]._set_source(var, *_SOURCES[-1][var])
    if _SCOPES and _SCOPES[-1]:
        _SCOPE_OF[-1][var] = tuple(_SCOPES[-1])


def _is_device(x):
    """Whether ``x`` names a device (``"cpu"``, ``"mps"``, ``"tpu"``, ``"cuda:0"``, a
    ``lumen.device``), not a dtype."""
    from lumen.device import device

    return isinstance(x, device) or (isinstance(x, str) and _device_type(x) in ("cpu", "mps", "cuda", "tpu", "meta"))


def _device_name(d):
    """Device ``d`` (a string or a ``lumen.device``) as ``"type:index"``: an
    unindexed one is device 0 of its type, as tensors' devices print."""
    from lumen.device import device

    d = device(d) if isinstance(d, str) else d
    return f"{d.type}:{d.index or 0}"


def _device_type(d):
    """The type of device ``d`` (``"cpu"`` of ``"cpu:0"``, a string or a
    ``lumen.device``), or None for none."""
    return None if d is None else str(d).split(":")[0]


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
        _SCOPES[-1].append((name, next(_CALLS)))
        _TRACES[-1]._set_scope(_SCOPES[-1])


def _exit_scope():
    """Close the trace's innermost ``record_function`` range, if one runs."""
    if _TRACES:
        _SCOPES[-1].pop()
        _TRACES[-1]._set_scope(_SCOPES[-1])


def _backward_scope(var, base):
    """The scope of value ``var``'s gradient ops: ``base`` (the ranges open
    where the backward runs), then each range ``var`` was traced in as
    ``name (backward)``, of the same call (one backward range per forward
    call)."""
    return tuple(base) + tuple((f"{name} (backward)", call) for name, call in _SCOPE_OF[-1].get(var, ()))


@contextlib.contextmanager
def _scope(ranges):
    """Trace inside the ``record_function`` ranges ``ranges`` (outermost
    first, each ``(name, call)``), the open ones back after."""
    saved = list(_SCOPES[-1])
    _SCOPES[-1][:] = ranges
    _TRACES[-1]._set_scope(_SCOPES[-1])
    try:
        yield
    finally:
        _SCOPES[-1][:] = saved
        _TRACES[-1]._set_scope(saved)


def producer(var):
    """The op of the current trace that computed value ``var``: ``(name,
    input vars, params)``, or None (an input)."""
    return _PRODUCERS[-1].get(var)


@contextlib.contextmanager
def tracing(graph, device=None, sources=None, tape=None, leaves=None, host=None):
    """Ops record into ``graph`` (a ``Graph``, or anything answering as one:
    a ``lumen.pallas`` kernel) inside: a trace of its own, on ``device``,
    its values' source lines in ``sources``, its nodes on ``tape`` (for
    ``.backward()`` of ``leaves``), its scalars' math on ``host`` (a
    ``_Host``, or None). Yields its random state."""
    tape = [] if tape is None else tape
    rng = {"state": None, "drawn": 0}
    _TRACES.append(graph)
    _SCOPES.append([])
    _SCOPE_OF.append({})
    _DEVICES.append(device)
    _RNG.append(rng)
    _SOURCES.append({} if sources is None else sources)
    _PRODUCERS.append({})
    _TAPES.append(tape)
    _BACKWARD.append((tape, [] if leaves is None else leaves))
    _HOSTS.append(host)
    try:
        yield rng
    finally:
        _TRACES.pop()
        _SCOPES.pop()
        _SCOPE_OF.pop()
        _DEVICES.pop()
        _RNG.pop()
        _SOURCES.pop()
        _PRODUCERS.pop()
        _TAPES.pop()
        _BACKWARD.pop()
        _HOSTS.pop()


def compiler_config():
    """The compiler settings of the device being traced for
    (``lumen.config.compiler.of``)."""
    return config.compiler.of(_DEVICES[-1] if _DEVICES and _DEVICES[-1] is not None else "cpu")


def current_graph():
    """The graph being traced into; raises outside ``lumen.compile``."""
    if not _TRACES:
        raise RuntimeError("lumen ops run only while tracing, inside a function passed to lumen.compile")
    return _TRACES[-1]


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


def _runtime(dynamic):
    """Which scalars are runtime: floats and numpy scalars, and the ints and
    bools at the slots ``(argument, path)`` in ``dynamic``."""

    def runtime(slot, value):
        """Whether the scalar at ``slot`` is a runtime input rather than baked in."""
        return type(value) not in (int, bool) or slot in dynamic

    return runtime


def _slots(args):
    """``(slot, value)`` for each scalar of ``args`` (``nn.is_scalar``), its
    slot ``(argument, path)``, in order."""
    return [((i, path), nn.at(a, path)) for i, a in enumerate(args) for path in nn.scalar_paths(a)]


def _runtime_slots(args, runtime):
    """``(slot, value)`` for each runtime scalar of ``args``, in order: the
    plan's scalar inputs."""
    return [(slot, v) for slot, v in _slots(args) if runtime(slot, v)]


def _scalar_dtype(device, value, portable=False):
    """The dtype of a runtime scalar of ``value`` on ``device``: a bool's,
    a numpy scalar's own, an int's int64, a float's float64; 64-bit types
    32-bit where the device has none (MPS's float64, TPU's 64-bit types),
    and float64 in a ``portable`` graph (the meta device's, ``make_graph``'s:
    compiled for any device)."""
    import numpy as np

    kind = "meta" if portable else _device_type(device)
    if isinstance(value, (bool, np.bool_)):
        return "bool"
    dtype = value.dtype.name if isinstance(value, np.generic) else "int64" if isinstance(value, int) else "float64"
    narrow = {"float64": kind in ("mps", "tpu", "meta"), "int64": kind == "tpu", "uint64": kind == "tpu"}
    return {"float64": "float32", "int64": "int32", "uint64": "uint32"}[dtype] if narrow.get(dtype) else dtype


def _plain(value):
    """A numpy scalar's Python value; any other scalar as it is."""
    return value.item() if hasattr(value, "item") and not isinstance(value, (int, float, bool)) else value


def _python(value, like):
    """``value`` (a 0-d tensor's) as a scalar of ``like``'s type."""
    (v,) = value.tolist()
    return type(like)(v)


class _Host:
    """The host part of a trace: the ops of 0-d values computed from runtime
    scalars and constants alone (a step count's ``t + 1``, a bias
    correction), recorded into ``graph`` to run on the CPU each call, before
    the device plan. A device op reading one of its values reads it as a
    scalar input of the device graph, passed by value (``imported``): the
    scalars' own inputs, or values it computes (``hoisted``)."""

    def __init__(self, device_graph, device):
        """A host trace feeding ``device_graph`` on ``device``."""
        self.graph = Graph()
        self.device_graph, self.device = device_graph, device
        self.imports, self.hoisted, self.constants = {}, [], {}

    def owns(self, x):
        """Whether ``x`` is a value of the host graph."""
        return isinstance(x, TracedTensor) and x.graph is self.graph

    def precreate(self, var, dtype, shape):
        """The device input for host value ``var``, created now."""
        t = TracedTensor(self.device_graph, self.device_graph.input(dtype, list(shape)), self.device)
        self.imports[var] = t
        return t

    def imported(self, x):
        """Host value ``x`` as a value of the device graph: its input."""
        t = self.imports.get(x.var)
        if t is None:
            t = self.precreate(x.var, x.dtype, x.shape)
            self.hoisted.append(x.var)
        t.weak = x.weak
        return t

    def replay(self, x):
        """Constant ``x`` of the device graph (``recipe``) recorded again in
        the host graph: its value there."""
        v = self.constants.get(x.var)
        if v is None:
            name, params, operands = x.recipe
            v = self.graph.apply(name, [self.replay(o) for o in operands], params)
            self.constants[x.var] = v
        return v

    def value(self, x):
        """``x`` (a host value or a constant) as a value of the host graph."""
        return x.var if self.owns(x) else self.replay(x)

    def bind(self, name, operands, params):
        """``name`` of ``operands`` recorded in the host graph if they are
        its values and constants, 0-d, and so is the result; else None."""
        if not any(self.owns(x) for x in operands):
            return None
        constant = lambda x: isinstance(x, TracedTensor) and (self.owns(x) or x.recipe is not None)  # noqa: E731
        if not all(constant(x) and not x.shape for x in operands):
            return None
        var = self.graph.apply(name, [self.value(x) for x in operands], params)
        if self.graph.type_of(var)[1]:
            return None
        return TracedTensor(self.graph, var, self.device)


def _trace(fn, args, device, runtime=_runtime(()), portable=False, split=False):
    """``fn`` traced on ``args``: the graph, and what ``fn`` returned (a
    traced tensor, or modules, lists and tuples of them: the structure the
    graph's outputs, its traced tensors in order, are returned in). The graph's inputs
    are the tensor arguments, then the runtime scalars (``_runtime_slots``
    of ``runtime``: 0-d inputs of ``_scalar_dtype`` (``portable``), weakly typed: each
    takes its tensor operand's dtype; a module's scalar field assigned in
    ``fn`` written back), then the weights of the module arguments
    (``_weights``), each a whole meta tensor, then, if ``fn`` draws random
    numbers (``lumen.random``), the random state (the seed and the stream's
    position, uint64 runtime scalars), and the host values device ops read
    (``hoisted``), in the order they are first read; the numbers it draws
    are returned, then the static scalars ``fn`` assigned (``(slot,
    value)``), then each input's kind and index in its kind (``layout``:
    ``arg``, ``scalar``, ``hoisted``, ``weight``, ``rng``).

    With ``split``, the scalars' math runs on the host (``_Host``): the host
    graph computes the hoisted values and the scalars ``fn`` assigns from
    scalars alone, from the scalars, each call; without (``make_graph``),
    in the one graph. Precision warnings point at the line computing the
    value they are about."""
    graph = Graph()
    traced = [TracedTensor(graph, graph.input(a.dtype, a.shape), device) if isinstance(a, Tensor) else a for a in args]
    runtime_slots = _runtime_slots(args, runtime)
    host = _Host(graph, device) if split else None
    kinds = {t.var: ("arg", k) for k, t in enumerate(t for t in traced if isinstance(t, TracedTensor))}
    inputs = {}
    for k, (slot, value) in enumerate(runtime_slots):
        dtype = _scalar_dtype(device, value, portable)
        owner = graph if host is None else host.graph
        inputs[slot] = TracedTensor(owner, owner.input(dtype, []), device)
        inputs[slot].weak = True
        kinds[inputs[slot].var if host is None else host.precreate(inputs[slot].var, dtype, ()).var] = ("scalar", k)
    weights = {}
    for t in _weights(args):
        weights[t.storage_id] = TracedTensor(graph, graph.input(t.dtype, t.shape), device)
        kinds[weights[t.storage_id].var] = ("weight", len(weights) - 1)

    def mapped(i, a):
        """Argument ``i`` with its module weights and runtime scalars swapped for their graph inputs."""
        if isinstance(a, TracedTensor):
            return a
        return nn.map_tensors(a, lambda t: weights.get(t.storage_id, t), lambda path, v: inputs.get((i, path), v))

    traced = [mapped(i, a) for i, a in enumerate(traced)]
    scalars = [inputs[slot] for slot, _ in runtime_slots]

    sources = {}
    arguments = [t for t in traced if isinstance(t, TracedTensor) and not t.weak]
    assigned = {sid: t.var for sid, t in weights.items()}
    assigned_scalars = [(t.graph, t.var) for t in scalars]
    assigned_arguments = [t.var for t in arguments]
    tape = []
    leaves = [t for t in traced if isinstance(t, TracedTensor) and not t.weak] + list(weights.values())

    with tracing(graph, device, sources, tape, leaves, host) as rng:
        out = fn(*traced)
        changed = []
        positions = {slot: k for k, (slot, _) in enumerate(runtime_slots)}
        for slot, value in _slots(args):
            i, path = slot
            if not path:
                continue
            now = nn.at(traced[i], path)
            k = positions.get(slot)
            if k is None:
                # A changed static scalar is reported so _settled retraces with it runtime, or pins it.
                if isinstance(now, TracedTensor) or type(now) is not type(value) or now != value:
                    changed.append((slot, now))
            elif now is not scalars[k]:
                dtype = _scalar_dtype(device, value, portable)
                now = _as_tensor(now, dtype, "a scalar field's value")
                scalars[k] = now if now.dtype == dtype else now.to(dtype)
        if host is not None:
            out = _tree_map(lambda o: host.imported(o) if host.owns(o) else o, out)
    if host is not None:
        kinds.update({host.imports[v].var: ("hoisted", j) for j, v in enumerate(host.hoisted)})
    kinds.update({t.var: ("rng", k) for k, t in enumerate(rng["state"] or ())})
    # Map each graph input to its kind and index, in the graph's input order.
    layout = [kinds[v] for v in graph.inputs()]

    outputs = _tree_leaves(out)
    if not isinstance(out, (TracedTensor, nn.Module, list, tuple)) or any(o.graph is not graph for o in outputs):
        raise TypeError(f"a compiled function must return tensors computed from its inputs, got {out!r}")
    # Values reassigned during the trace become extra outputs after the returned ones.
    written = [k for k, (sid, t) in enumerate(weights.items()) if t.var != assigned[sid]]
    written_scalars = [k for k, t in enumerate(scalars) if (t.graph, t.var) != assigned_scalars[k]]
    on_host = [host is not None and (host.owns(scalars[k]) or scalars[k].recipe is not None) for k in written_scalars]
    host_written = [k for k, h in zip(written_scalars, on_host) if h]
    written_scalars = [k for k, h in zip(written_scalars, on_host) if not h]
    written_slots = [runtime_slots[k][0] for k in written_scalars]
    written_arguments = [k for k, t in enumerate(arguments) if t.var != assigned_arguments[k]]
    graph.set_outputs(
        [o.var for o in outputs]
        + [list(weights.values())[k].var for k in written]
        + [scalars[k].var for k in written_scalars]
        + [arguments[k].var for k in written_arguments]
    )
    host_part = (None, 0, [])
    if host is not None and (host.hoisted or host_written):
        host.graph.set_outputs(host.hoisted + [host.value(scalars[k]) for k in host_written])
        host.graph.prune()
        host_part = (host.graph, len(host.hoisted), [runtime_slots[k][0] for k in host_written])
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
    return graph, out, (written, written_slots, written_arguments, host_part), drawn, changed, layout


def _lift(x):
    """``x`` as an operand of the graph being traced; a tensor the function
    closes over is an error (it would be invisible to the graph)."""
    if isinstance(x, Tensor) and _TRACES:
        raise TypeError(
            f"a compiled function cannot close over a tensor ({x.device}, shape {x.shape}): pass data as an argument, and weights in a lumen.nn.Module argument"
        )
    return x


def _signature(args, runtime):
    """What a plan for ``args`` depends on: the tensors' types, the
    modules' and containers' structure (``nn.structure``: a runtime
    scalar's type, a static one's value), which weights are shared, and the
    other arguments' values."""
    ids = [t.storage_id for a in args if isinstance(a, nn.Module) for _, t in nn._leaves(a, "")]
    shared = tuple(ids.index(i) for i in ids)

    def of(i, a):
        """Argument ``i``'s part of the signature."""
        if isinstance(a, Tensor):
            return (a.dtype, tuple(a.shape))
        if nn.is_scalar(a) or isinstance(a, (nn.Module, list, tuple)):
            return nn.structure(a, lambda path, v: runtime((i, path), v))
        return ("static", a)

    return tuple(of(i, a) for i, a in enumerate(args)) + (shared,)


def _settled(fn, args, device, dynamic, pinned, proven, portable=False, split=True):
    """``fn`` traced on ``args`` (``_trace``) with the ints and bools at the
    slots in ``dynamic`` runtime, traced again until it settles: one ``fn``
    assigns is made runtime, written back each call; one that cannot be
    (used as a shape or a condition, which a runtime scalar cannot be) is
    ``pinned`` static, its new value baked in. ``proven``: the slots traced
    runtime before. With ``split``, the scalars' math runs on the host
    (``_trace``)."""
    while True:
        try:
            result = _trace(fn, args, device, _runtime(dynamic), portable, split)
        except Exception:
            fresh = dynamic - proven
            if not fresh:
                raise
            dynamic -= fresh
            pinned |= fresh
            continue
        need = {slot for slot, _ in result[4] if slot not in pinned}
        if need:
            dynamic |= need
            continue
        for (i, path), now in result[4]:
            if isinstance(now, TracedTensor):
                where = ".".join(map(str, path))
                raise TypeError(
                    f"argument {i}'s {where} is static (a shape or a condition): it cannot be assigned a traced value"
                )
        proven |= dynamic
        return result


_flags = [None, None]


def _compiler_flags():
    """``repr(config.compiler)``, the flags plans are compiled with."""
    version = config.compiler._version
    if _flags[0] != version:
        _flags[:] = [version, repr(config.compiler)]
    return _flags[1]


def _device(args, device):
    """Where a function of ``args`` runs: ``device``, or the first tensor
    argument with data's, or the meta device if all are meta (nothing
    runs), or the CPU."""
    if device is not None:
        return _device_name(device)
    tensors = [a for a in args if isinstance(a, Tensor)]
    data = [a for a in tensors if _device_type(a.device) != "meta"]
    return str(data[0].device) if data else ("meta:0" if tensors else "cpu:0")


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
    call's signature, profiled, as an HTML page (``lumen.viz``);
    ``compiled.dump_graph(path, *args)``, those of ``args``' signature.
    ``compiled.dump_code(directory)`` writes the code of every kernel a call
    of that signature runs, in the order it runs them, a file each
    (``lumen.viz.dump_code``).

    With ``lumen.config.compiler.mps.neural_engine``, an MPS plan runs the large
    float16 dots of the weights the function does not write (and the
    float16 work after them) on the Apple Neural Engine, their values baked
    into Core ML programs: compiled again once a weight is written
    (``copy_``, or by another compiled function)."""
    compile_device = device
    plans = {}
    latest = []
    named = None if device is None else _device_name(device)
    dynamic, pinned, proven, seen = set(), set(), set(), {}

    def prepare(args):
        """The plan for ``args`` and its inputs: the graph, the plan, its
        workspace, what ``fn`` returns (its outputs' structure), and the plan's
        inputs (the arguments, then the weights placed, and their blocks)."""
        target = named or _device(args, None)
        # An int or bool seen with two values across calls is made runtime instead of recompiling per value.
        for slot, value in _slots(args):
            if type(value) in (int, bool) and slot not in dynamic and slot not in pinned:
                if seen.setdefault(slot, (type(value), value)) != (type(value), value):
                    dynamic.add(slot)
        key = _signature(args, _runtime(dynamic)), target, _compiler_flags()
        result = None
        if key not in plans:
            result = _settled(fn, args, target, dynamic, pinned, proven)
            key = _signature(args, _runtime(dynamic)), target, _compiler_flags()
        if result is not None:
            plans[key] = (*result, [])
        graph, out, written_all, drawn, _, layout, entries = plans[key]
        host_graph, n_hoisted, host_slots = written_all[3]
        runtime_slots = _runtime_slots(args, _runtime(dynamic))
        raw = [Tensor.full([], _plain(v), _scalar_dtype(target, v)) for _, v in runtime_slots]
        pending[:] = []
        hoisted = []
        if host_graph is not None:
            if key not in hosts:
                hosts[key] = Plan(host_graph, "cpu")
            values = hosts[key].run(raw)
            hoisted, pending[:] = values[:n_hoisted], list(zip(host_slots, values[n_hoisted:]))
        from lumen import random

        weights = _weights(args)
        of = {
            "arg": [a for a in args if isinstance(a, Tensor)],
            "scalar": raw,
            "hoisted": hoisted,
            "weight": weights,
            "rng": random._state() if drawn is not None else [],
        }
        logical = [of[kind][k] for kind, k in layout]
        latest[:] = [key]
        if _device_type(target) == "meta":
            if not entries:
                entries.append((Plan(graph, "meta"), None))
            return graph, entries[0][0], None, out, logical
        params = [i for i, (kind, _) in enumerate(layout) if kind == "weight"]
        scalars = [i for i, (kind, _) in enumerate(layout) if kind in ("scalar", "hoisted", "rng")]

        def inputs_for(plan):
            """The plan's inputs for this call, or None if a packed weight block cannot be read in place."""
            blocks = []
            for (positions, dimension), regions in zip(plan.packed, plan.interleaved):
                block = _pack([logical[i] for i in positions], dimension, target, regions)
                if block is None:
                    return None
                blocks.append(block)
            return [v._placed(target) if kind == "weight" else v for (kind, _), v in zip(layout, logical)] + blocks

        # Reuse the first cached plan whose packed weight blocks this call's weights can fill.
        for plan, workspace in entries:
            inputs = inputs_for(plan)
            if inputs is not None:
                return graph, plan, workspace, out, inputs
        packable = params if not entries else []
        written, written_scalars, written_arguments, _ = written_all
        returned = len(graph.outputs()) - len(written) - len(written_scalars) - len(written_arguments)
        donate = [(params[w], returned + j) for j, w in enumerate(written)]
        plan = Plan(graph, target, donate_into=donate, parameters=params, packable=packable, scalars=scalars)
        workspace = Tensor.empty([plan.workspace_bytes], "uint8", target, alignment=plan.workspace_alignment)
        entries.append((plan, workspace))
        inputs = inputs_for(plan)
        # Weights that cannot be packed in place get a plan without packing.
        if inputs is None:
            plan = Plan(graph, target, donate_into=donate, parameters=params, scalars=scalars)
            entries[-1] = plan, workspace
            inputs = inputs_for(plan)
        return graph, plan, workspace, out, inputs

    baked = {}

    hosts, pending = {}, []

    fast = {}

    @functools.wraps(fn)
    def compiled(*args):
        """Run ``fn``'s plan for ``args``, compiling one for a new signature; see ``compile``."""
        # Fast path: plain tensor arguments whose signature already has a plan with nothing to write back.
        if all(type(a) is Tensor for a in args):
            hit = fast.get((tuple([a._signature for a in args]), _compiler_flags()))
            if hit is not None:
                plan, workspace, out, to_host, key = hit
                latest[:] = [key]
                if to_host is None:
                    return plan.run_in(workspace, list(args))[0]
                outputs = iter(zip(plan.run_in(workspace, list(args)), to_host))

                def returned(_):
                    """The next plan output, copied to the host if its traced tensor was on the CPU."""
                    value, host = next(outputs)
                    return value.cpu() if host else value

                return _tree_map(returned, out)
        graph, plan, workspace, out, inputs = prepare(args)
        ran = not (workspace is None or any(isinstance(a, Tensor) and _device_type(a.device) == "meta" for a in args))
        if ran and plan.neural_engine:
            versions = [w._version for w in _weights(args)]
            if baked.get(id(plan)) != versions:
                plan._invalidate_neural_engine()
                baked[id(plan)] = versions
        if not ran:
            outputs = [
                Tensor.empty(list(shape), dtype, "meta") for dtype, shape in map(graph.type_of, graph.outputs())
            ]
        else:
            outputs = plan.run_in(workspace, inputs)
        outputs = iter(outputs)

        to_host = []

        def result_of(t):
            """The next plan output for traced output ``t``, copied to the host when ``t`` is a CPU value."""
            value = next(outputs)
            host = ran and _device_type(t.device) == "cpu" and _device_type(value.device) != "cpu"
            to_host.append(host)
            return value.cpu() if host else value

        result = _tree_map(result_of, out)
        if ran and all(type(a) is Tensor for a in args):
            written, written_scalars, written_arguments, host = plans[latest[0]][2]
            drawn = plans[latest[0]][3]
            plain = host[0] is None and not (written or written_scalars or written_arguments or drawn is not None)
            if plain and not plan.neural_engine and len(inputs) == len(args):
                signature = tuple([a._signature for a in args])
                lone = isinstance(out, TracedTensor) and to_host == [False]
                fast[signature, _compiler_flags()] = (plan, workspace, out, None if lone else to_host, latest[0])
        # Write assigned weights, scalars and arguments back, and advance the random stream.
        if ran:
            weights = _weights(args)
            written, written_scalars, written_arguments, _ = plans[latest[0]][2]
            target = latest[0][1]
            for k, value in zip(written, outputs):
                w = weights[k]
                memory = w._placed(target) if w._is_parameter else w
                if value.storage_id != memory.storage_id:
                    w.copy_(value)
                elif w._is_parameter:
                    w._mark_written()
            for (i, path), value in zip(written_scalars, outputs):
                nn.assign(args[i], path, _python(value, nn.at(args[i], path)))
            for (i, path), value in plans[latest[0]][4]:
                nn.assign(args[i], path, value)
            for (i, path), value in pending:
                nn.assign(args[i], path, _python(value, nn.at(args[i], path)))
            arguments = [a for a in args if isinstance(a, Tensor)]
            for k, value in zip(written_arguments, outputs):
                arguments[k].copy_(value)
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
        graph, plan, target, inputs, workspace, scalars = planned("dump_graph", args, device)
        from lumen import viz

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

    def dump_code(directory, *args, device=None):
        """Write the code of every kernel the plan for ``args`` (default: the latest call's) runs to ``directory``."""
        graph, plan, target, inputs, workspace, _ = planned("dump_code", args, device)
        from lumen import viz

        return viz.dump_code(graph, plan, lambda: plan.run_in(workspace, inputs), target, directory)

    def planned(name, args, device):
        """Graph, a fresh plan on ``device``, ones-filled inputs and a workspace for ``args``' signature, for ``name``."""
        if args:
            prepare(args)
        elif not latest:
            raise RuntimeError(f"{name}: call {fn.__name__} first, or pass it arguments to trace")
        ((_, target, _),) = latest
        graph, _, _, _, _, layout, _ = plans[latest[0]]
        target = _device_name(device or compile_device or (target if _device_type(target) != "meta" else "cpu"))
        types = [graph.type_of(v) for v in graph.inputs()]
        params = [i for i, (kind, _) in enumerate(layout) if kind == "weight"]
        scalars = [i for i, (kind, _) in enumerate(layout) if kind in ("scalar", "hoisted", "rng")]
        plan = Plan(graph, target, parameters=params, packable=params, scalars=scalars)
        inputs = [Tensor.ones(list(shape), dtype, target) for dtype, shape in types]
        for positions, dimension in plan.packed:
            dtype, shape = types[positions[0]]
            shape = list(shape)
            shape[dimension] = sum(types[i][1][dimension] for i in positions)
            inputs.append(Tensor.ones(shape, dtype, target))
        workspace = Tensor.empty([plan.workspace_bytes], "uint8", target, alignment=plan.workspace_alignment)
        return graph, plan, target, inputs, workspace, scalars

    compiled.dump_graph = dump_graph
    compiled.dump_code = dump_code
    return compiled


def make_graph(fn):
    """A function returning the graph ``fn`` traces to on the given
    arguments (``jax.make_jaxpr``), which may be meta tensors; ``print`` it
    to read it. Its runtime scalars are portable (``_scalar_dtype``): the
    graph compiles for any device."""

    @functools.wraps(fn)
    def graph(*args):
        """The graph ``fn`` traces to on ``args``."""
        return _settled(fn, args, _device(args, None), set(), set(), set(), portable=True, split=False)[0]

    return graph


def _is_float(dtype):
    """Whether ``dtype`` names a floating-point dtype."""
    return "float" in dtype


def _accum_dtype(dtype):
    """The dtype a matmul of ``dtype`` accumulates in: float32 for floats,
    or their own if wider (float64); integers their own."""
    return dtype if not _is_float(dtype) or dtype == "float64" else "float32"


def _require_float(x, op):
    """``x``, or a TypeError naming ``op`` if it is not floating-point."""
    if not _is_float(x.dtype):
        raise TypeError(f"{op} needs a floating-point tensor, got {x.dtype}: convert it with .to(dtype)")
    return x


def _common_dtype(op, operands):
    """The dtype the tensors among ``operands`` share (weakly typed ones,
    runtime scalars, take the others'; of weakly typed ones alone, the
    widest float's, else int's, else bool's: ``step * log(beta)``)."""
    tensors = [x for x in map(_lift, operands) if isinstance(x, TracedTensor)]
    dtypes = sorted({x.dtype for x in tensors if not x.weak} or {x.dtype for x in tensors})
    if not dtypes:
        for x in operands:
            if hasattr(x, "_not_an_operand"):
                raise TypeError(x._not_an_operand(op))
        raise TypeError(f"{op} needs a tensor operand")
    if len(dtypes) > 1 and all(x.weak for x in tensors):
        width = lambda d: int("".join(c for c in d if c.isdigit()) or 8)  # noqa: E731
        return max(dtypes, key=lambda d: ("float" in d, "int" in d, width(d)))
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


def _is_operand(x):
    """Whether ``x`` can be an elementwise operand: a traced tensor or a Python scalar."""
    return isinstance(x, (TracedTensor, bool, int, float))


def _as_tensor(x, dtype, op):
    """``x`` (a traced tensor of ``dtype``, or a Python scalar of its kind)
    as a traced tensor."""
    x = _lift(x)
    if isinstance(x, TracedTensor) and x.weak and x.dtype != dtype:
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
    """The NumPy broadcast of ``shapes``, aligned at the trailing dimensions."""
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
    """``x`` broadcast to ``shape`` by adding leading dimensions and expanding size-1 ones."""
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
    """Elementwise minimum; floats as ``-max(-x, -y)`` to keep ``max``'s NaN handling."""
    if _is_float(x.dtype):
        return prims.neg(prims.max(prims.neg(x), prims.neg(y)))
    return prims.select(prims.lt(y, x), y, x)


def _le(x, y):
    """Elementwise ``x <= y``."""
    less = prims.lt(x, y)
    return prims.select(less, less, prims.eq(x, y))


def _ne(x, y):
    """Elementwise ``x != y``, as ``(x == y) == False``."""
    return prims.eq(prims.eq(x, y), prims.full(x.shape, False, "bool"))


def _binary(prim, reflected=False):
    """An operator method: ``prim`` on the operands, promoted and broadcast."""

    def op(self, other):
        """``prim`` of ``self`` and ``other`` (swapped if ``reflected``), or NotImplemented for a non-operand."""
        other = _lift(other)
        if not _is_operand(other):
            return NotImplemented
        return _elementwise(prim, other, self) if reflected else _elementwise(prim, self, other)

    return op


def _floor_mod(x, y):
    """``x % y`` with Python's (and torch's) sign: the divisor's."""
    x, y = _operands("rem", x, y)
    r = prims.rem(x, y)
    zero = prims.full(r.shape, 0, r.dtype)
    fix = _and(_ne(r, zero), _ne(prims.lt(r, zero), prims.lt(y, zero)))
    return prims.select(fix, prims.add(r, y), r)


def _floor_div(x, y):
    """``x // y``: the quotient rounded down (``torch.floor_divide``)."""
    x, y = _operands("div", x, y)
    if _is_float(x.dtype):
        return prims.floor(prims.div(x, y))
    q = prims.div(x, y)
    zero = prims.full(q.shape, 0, q.dtype)
    inexact = _ne(prims.rem(x, y), zero)
    fix = _and(inexact, _ne(prims.lt(x, zero), prims.lt(y, zero)))
    return prims.select(fix, prims.sub(q, prims.full(q.shape, 1, q.dtype)), q)


def _and(x, y):
    """Elementwise logical or bitwise and."""
    return prims.and_(x, y)


def _right_shift(x, y):
    """``x >> y``: arithmetic for signed integers, logical for unsigned."""
    x, y = _operands("shift_right", x, y)
    shift = prims.shift_right_logical if x.dtype.startswith("uint") else prims.shift_right_arithmetic
    return shift(x, y)


def _power(x, y):
    """``x ** y``: a float's power (of floats, or integers: ``lax.pow``); an
    integer's to a non-negative Python int exponent, by multiplications
    (``lax.integer_pow``)."""
    if isinstance(y, int) and not isinstance(y, bool) and not _is_float(x.dtype) and y >= 0:
        result, base = prims.full(x.shape, 1, x.dtype), x
        while y:
            if y & 1:
                result = prims.mul(result, base)
            base, y = prims.mul(base, base), y >> 1
        return result
    if isinstance(y, TracedTensor) and _is_float(x.dtype) and not _is_float(y.dtype) and y.dtype != "bool":
        shape = _broadcast_shapes(x.shape, y.shape)
        return prims.pow(_broadcast_to(x, shape), _broadcast_to(y, shape))
    return _elementwise(prims.pow, x, y)


def _true_div(x, y):
    """``x / y`` (floats only); a constant or weak divisor is applied as a multiply by its reciprocal."""
    # A static divisor folds into a multiply by its reciprocal.
    if isinstance(y, (int, float)) and not isinstance(y, bool) and y != 0:
        return _elementwise(prims.mul, _require_float(x, "true division (/)"), 1 / y)
    if isinstance(y, TracedTensor) and y.weak and not getattr(x, "weak", True):
        r = _true_div(1.0, y)
        r.weak = True
        return _elementwise(prims.mul, _require_float(x, "true division (/)"), r)
    # The result is weak only if every tensor operand is.
    tensors = [v for v in (x, y) if isinstance(v, TracedTensor)]
    x, y = _operands("div", x, y)
    out = prims.div(_require_float(x, "true division (/)"), y)
    out.weak = bool(tensors) and all(v.weak for v in tensors)
    return out


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
        elif op[0] == "dynamic_slice":
            shape = list(op[2])
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
    if op[0] == "dynamic_slice":
        return prims.dynamic_slice(x, op[1], op[2])
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

    __slots__ = (
        "graph",
        "_var",
        "dtype",
        "shape",
        "weak",
        "grad",
        "_base",
        "_ops",
        "_seen",
        "_version",
        "device",
        "recipe",
    )

    def __init__(self, graph, var, device=None):
        """A traced value ``var`` of ``graph`` on ``device`` (default: the trace's)."""
        self.graph = graph
        self._var = var
        self._base, self._ops, self._seen, self._version = None, (), 0, 0
        self.weak = False
        self.grad = None
        self.recipe = None
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
        """The value's graph variable, dtype and shape."""
        return f"TracedTensor(%{self.var}, dtype={self.dtype}, shape={list(self.shape)})"

    def __bool__(self):
        """Raises: a traced value has no data to branch on."""
        raise TypeError("a traced tensor has no value: control flow cannot depend on tensor data inside lumen.compile")

    __hash__ = object.__hash__

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    def dim(self):
        """Number of dimensions (torch's ``dim()``)."""
        return self.ndim

    def size(self, dim=None):
        """The shape, or the size of dimension ``dim``."""
        return self.shape if dim is None else self.shape[_dim(dim, self.ndim)]

    def numel(self):
        """Number of elements."""
        return math.prod(self.shape)

    def __len__(self):
        """Size of the first dimension; raises for a 0-d tensor."""
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

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
        """``self / other`` (float division)."""
        return _true_div(self, other) if _is_operand(other) else NotImplemented

    def __rtruediv__(self, other):
        """``other / self`` (float division)."""
        return _true_div(other, self) if _is_operand(other) else NotImplemented

    def __neg__(self):
        """Elementwise negation."""
        return prims.neg(self)

    __and__ = _binary(prims.and_)
    __rand__ = _binary(prims.and_, reflected=True)
    __or__ = _binary(prims.or_)
    __ror__ = _binary(prims.or_, reflected=True)
    __xor__ = _binary(prims.xor)
    __rxor__ = _binary(prims.xor, reflected=True)
    __lshift__ = _binary(prims.shift_left)
    __rlshift__ = _binary(prims.shift_left, reflected=True)
    __rshift__ = _binary(_right_shift)
    __rrshift__ = _binary(_right_shift, reflected=True)
    __mod__ = _binary(_floor_mod)
    __rmod__ = _binary(_floor_mod, reflected=True)
    __floordiv__ = _binary(_floor_div)
    __rfloordiv__ = _binary(_floor_div, reflected=True)

    def __pow__(self, other):
        """``self ** other``."""
        return _power(self, other) if _is_operand(other) else NotImplemented

    def __rpow__(self, other):
        """``other ** self``."""
        return _elementwise(prims.pow, other, self) if _is_operand(other) else NotImplemented

    def __invert__(self):
        """Logical not for bools, bitwise not for integers."""
        return prims.not_(self)

    def __abs__(self):
        """Elementwise absolute value."""
        return prims.abs(self)

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
        base = self._base
        if base is None:
            self.graph, self._var = value.graph, value.var
            self.recipe = value.recipe
            self._version += 1
        else:
            base._var = _scatter(base, self._ops, value).var
            base._version += 1
            self._var, self._seen = value.var, base._version
        return self

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
        """In-place ``self += alpha * other``."""
        return self._assign(self + (other if alpha == 1 else other * alpha), "add_")

    def sub_(self, other, alpha=1):
        """In-place ``self -= alpha * other``."""
        return self._assign(self - (other if alpha == 1 else other * alpha), "sub_")

    def mul_(self, other):
        """In-place ``self *= other``."""
        return self._assign(self * other, "mul_")

    def div_(self, other):
        """In-place ``self /= other``."""
        return self._assign(self / other, "div_")

    def neg_(self):
        """In-place negation."""
        return self._assign(-self, "neg_")

    def exp_(self):
        """In-place ``exp`` of a float tensor."""
        return self._assign(prims.exp(_require_float(self, "exp_")), "exp_")

    def log_(self):
        """In-place natural log of a float tensor."""
        return self._assign(prims.log(_require_float(self, "log_")), "log_")

    def sqrt_(self):
        """In-place square root of a float tensor."""
        return self._assign(prims.sqrt(_require_float(self, "sqrt_")), "sqrt_")

    def tanh_(self):
        """In-place ``tanh`` of a float tensor."""
        return self._assign(prims.tanh(_require_float(self, "tanh_")), "tanh_")

    def sigmoid_(self):
        """In-place logistic sigmoid of a float tensor."""
        return self._assign(prims.logistic(_require_float(self, "sigmoid_")), "sigmoid_")

    def relu_(self):
        """In-place ``max(self, 0)``."""
        return self._assign(_elementwise(prims.max, self, 0), "relu_")

    def clamp_(self, min=None, max=None):
        """In-place clamp to ``[min, max]``; either bound may be None but not both."""
        if min is None and max is None:
            raise RuntimeError("clamp_: at least one of 'min' or 'max' must not be None")
        value = self if min is None else _elementwise(prims.max, self, min)
        value = value if max is None else _elementwise(_min, value, max)
        return self._assign(value, "clamp_")

    def clamp_min_(self, min):
        """In-place lower clamp."""
        return self.clamp_(min=min)

    def clamp_max_(self, max):
        """In-place upper clamp."""
        return self.clamp_(max=max)

    def uniform_(self, from_=0.0, to=1.0):
        """Fill it with uniform random numbers in ``[from_, to)``
        (``Tensor.uniform_``, its ``from`` here ``from_``), drawn from the
        generator (``lumen.random``): a weight so filled inside a compiled
        function is written back after the call (an initialization)."""
        from lumen import random

        return self._assign(random._uniform(self.shape, self.dtype, from_, to), "uniform_")

    def zero_(self):
        """Fill it with zeros (False for bool)."""
        return self.fill_(False if self.dtype == "bool" else 0)

    def fill_(self, value):
        """Fill it with the 0-d ``value``, a scalar or tensor of its dtype."""
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

    def __iadd__(self, other):
        """``self += other``, in place."""
        return self.add_(other)

    def __isub__(self, other):
        """``self -= other``, in place."""
        return self.sub_(other)

    def __imul__(self, other):
        """``self *= other``, in place."""
        return self.mul_(other)

    def __itruediv__(self, other):
        """``self /= other``, in place."""
        return self.div_(other)

    def __setitem__(self, key, value):
        """``x[key] = value``: the indexed view of it assigned ``value`` (a
        Python scalar, or a tensor of its dtype broadcast to the view)."""
        view = self[key]
        base = view._base
        # Skip a self-assignment of the very view being written (``x[k] = x[k]`` after an in-place op on it).
        if isinstance(value, TracedTensor) and value is not view and base is not None:
            if value._base is base and value._ops == view._ops and value._seen == base._version:
                return
        value = _as_tensor(value, self.dtype, "index assignment")
        if value.dtype != self.dtype:
            raise TypeError(f"index assignment: got a {value.dtype} value for a {self.dtype} tensor")
        shape = value.shape
        # Drop leading size-1 dims of the value so it can broadcast to a lower-rank view, as torch does.
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
        from lumen.autograd.transforms import backward

        tape, leaves = _BACKWARD[-1]

        if gradient is None:
            if self.shape != () or not _is_float(self.dtype):
                raise RuntimeError(f"backward: a gradient is needed for a non-scalar output, got {self!r}")

            gradient = prims.full((), 1.0, self.dtype)

        for leaf, g in zip(leaves, backward(self, gradient, list(tape), leaves)):
            if g is not None:
                if leaf.grad is not None:
                    g = leaf.grad + g

                leaf.grad = g

    def __matmul__(self, other):
        """Raises: matmul needs explicit dtypes, through ``lumen.functional.matmul``."""
        raise TypeError("@ has no dtypes: use lumen.functional.matmul(x, y, accum_dtype, output_dtype)")

    def __rmatmul__(self, other):
        """Raises: matmul needs explicit dtypes, through ``lumen.functional.matmul``."""
        raise TypeError("@ has no dtypes: use lumen.functional.matmul(x, y, accum_dtype, output_dtype)")

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
        kind = _device_type(device)
        if kind == "cpu":
            out = out.cpu()
        elif kind not in (None, "meta") and _device_type(out.device) == "cpu":
            out = prims.to_device(out)
        return out

    def cpu(self):
        """It on the host (``Tensor.cpu``): itself if it is there, else copied
        there (``prims.to_host``: the plan waits for it and copies it). The
        ops reading it run on the host, as PyTorch runs ops on CPU tensors on
        the CPU (reading it and a device value in one op raises); returned,
        it is a CPU tensor."""
        if _device_type(self.device) == "cpu":
            return self
        out = prims.to_host(self)
        out.weak = self.weak
        return out

    def type_as(self, other):
        """It cast to ``other``'s dtype."""
        return self.to(other.dtype)

    def float(self):
        """It cast to float32."""
        return self.to("float32")

    def double(self):
        """It cast to float64."""
        return self.to("float64")

    def half(self):
        """It cast to float16."""
        return self.to("float16")

    def bfloat16(self):
        """It cast to bfloat16."""
        return self.to("bfloat16")

    def int(self):
        """It cast to int32."""
        return self.to("int32")

    def long(self):
        """It cast to int64."""
        return self.to("int64")

    def bool(self):
        """It cast to bool."""
        return self.to("bool")

    def reshape(self, *shape):
        """It reshaped to ``shape`` (one ``-1`` inferred), a view of it when torch's strides allow one."""
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
        # Only a reshape torch could do without copying stays a view that sees later in-place writes.
        if _reshape_strides(self.shape, _strides(self), shape) is None:
            return out
        return _view(self, out, ("reshape", tuple(shape)))

    view = reshape

    def flatten(self, start_dim=0, end_dim=-1):
        """Dimensions ``start_dim..=end_dim`` merged into one."""
        if self.ndim == 0:
            return self.reshape(1)
        start, end = _dim(start_dim, self.ndim), _dim(end_dim, self.ndim)
        return self.reshape(self.shape[:start] + (-1,) + self.shape[end + 1 :])

    def unsqueeze(self, dim):
        """A size-1 dimension inserted at ``dim``."""
        dim = _dim(dim, self.ndim + 1)
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim=None):
        """Size-1 dimensions removed: those in ``dim``, or all if None."""
        dims = _dims(dim, self.ndim)
        return self.reshape(tuple(n for d, n in enumerate(self.shape) if not (d in dims and n == 1)))

    def permute(self, *dims):
        """A view with its dimensions reordered to ``dims``."""
        dims = [_dim(d, self.ndim) for d in _sizes(dims)]
        if sorted(dims) != list(range(self.ndim)):
            raise RuntimeError(f"permute: {dims} is not a permutation of the dimensions of a {self.ndim}-d tensor")
        return self if dims == sorted(dims) else _view(self, prims.transpose(self, dims), ("permute", tuple(dims)))

    def transpose(self, dim0, dim1):
        """A view with dimensions ``dim0`` and ``dim1`` swapped."""
        dims = list(range(self.ndim))
        a, b = _dim(dim0, self.ndim), _dim(dim1, self.ndim)
        dims[a], dims[b] = dims[b], dims[a]
        return self.permute(dims)

    def t(self):
        """The transpose of a 2-d tensor; a 0-d or 1-d one as it is."""
        if self.ndim > 2:
            raise RuntimeError(f"t() expects a tensor with <= 2 dimensions, but self is {self.ndim}D")
        return self.transpose(0, -1) if self.ndim == 2 else self

    @property
    def mT(self):
        """Its last two dimensions swapped."""
        return self.transpose(-2, -1)

    def expand(self, *sizes):
        """It broadcast to ``sizes`` (``-1`` keeps a dimension), adding leading dimensions as needed."""
        sizes = _sizes(sizes)
        lead = len(sizes) - self.ndim
        if lead < 0:
            raise RuntimeError(
                f"expand: the number of sizes ({len(sizes)}) must be at least the tensor's rank ({self.ndim})"
            )
        shape = tuple(self.shape[i - lead] if n == -1 and i >= lead else n for i, n in enumerate(sizes))
        return _broadcast_to(self, shape)

    def expand_as(self, other):
        """It broadcast to ``other``'s shape."""
        return self.expand(other.shape)

    def broadcast_to(self, shape):
        """It broadcast to ``shape``."""
        return self.expand(shape)

    def _slice(self, starts, limits):
        """A view of the static block ``[starts, limits)``; itself if that is the whole tensor."""
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
        # Expand the one ``...`` into full slices so the key has one entry per dimension.
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
        # Integer indices kept their dimension as size 1 in the slice; reshape drops them.
        return out if tuple(shape) == out.shape else out.reshape(shape)

    def narrow(self, dim, start, length):
        """The ``length`` elements of dimension ``dim`` from ``start``: an
        int, or a 0-d integer tensor read when it runs (clamped so the
        elements are inside, as ``prims.dynamic_slice``)."""
        d = _dim(dim, self.ndim)
        n = self.shape[d]
        if isinstance(start, TracedTensor):
            starts, sizes = [0] * self.ndim, list(self.shape)
            starts[d], sizes[d] = start, length
            out = prims.dynamic_slice(self, starts, sizes)
            return _view(self, out, ("dynamic_slice", tuple(starts), tuple(sizes)))
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
