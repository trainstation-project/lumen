"""The transformations (JAX: ``jax/_src/interpreters/ad.py``): forward
mode (``jvp``), linearize, transpose (``backward_pass``), and reverse mode
(``vjp``, ``grad``, ``value_and_grad``)."""

import functools

from lumen.autograd import attention, dots, rules
from lumen.autograd.core import (
    UndefinedPrimal,
    _accumulate,
    _full,
    _is_float,
    _value,
    primitive_jvps,
    primitive_transposes,
)
from lumen.autograd.function import _FUNCTION, _function_jvp, _function_transpose
from lumen.graph import prims, tracer
from lumen.graph.tracer import TracedTensor


def _record(fn, *args):
    """``fn(*args)``, and the nodes it bound."""
    tape = []
    tracer._TAPES.append(tape)

    try:
        return fn(*args), tape
    finally:
        tracer._TAPES.pop()


def _jvp_tape(tape, tangents, linearizing=False):
    """Each node of ``tape``'s JVP rule, given ``tangents`` (var -> its
    tangent's var), which it extends with theirs. ``linearizing``: a
    :class:`Function`'s tangents are placeholders, outputs of one linear
    node its ``backward`` transposes."""

    graph = tracer.current_graph()

    for name, inputs, params, out in tape:
        ts = [tangents.get(v) for v in inputs]

        if name == _FUNCTION:
            if any(t is not None for t in ts):
                _function_jvp(params, ts, out, tangents, linearizing)

            continue

        if all(t is None for t in ts) or not _is_float(graph.type_of(out)[0]):
            continue

        primitive = getattr(prims, name, None)
        if name == "custom_call":
            raise NotImplementedError(f"custom op {params['op']} is not differentiable: its function is opaque")
        if primitive not in primitive_jvps:
            raise NotImplementedError(f"{name} has no JVP rule")

        ts = [_value(t) if t is not None else None for t in ts]

        t = primitive_jvps[primitive]([_value(v) for v in inputs], ts, _value(out), **params)
        if t is not None:
            tangents[out] = t.var


def _check(fn_name, args):
    if not tracer._TRACES:
        raise RuntimeError(f"lumen.{fn_name} runs while tracing: call it inside a function passed to lumen.compile")

    leaves = tracer._tree_leaves(args)
    if not leaves:
        raise TypeError(f"lumen.{fn_name}: nothing to differentiate: the arguments have no tensors")

    return leaves


def jvp(fn, primals, tangents):
    """``fn(*primals)`` and its derivative along ``tangents`` (forward
    mode, ``jax.jvp``): each a tree (tensors, modules, lists, tuples) of
    ``primals``' structure; the result's tangent is of its structure."""

    leaves = _check("jvp", primals)
    tangent_leaves = tracer._tree_leaves(tangents)

    if len(tangent_leaves) != len(leaves):
        raise ValueError(f"jvp: {len(leaves)} primals, {len(tangent_leaves)} tangents")

    out, tape = _record(fn, *primals)
    tangent = {}

    for p, t in zip(leaves, tangent_leaves):
        if (t.dtype, t.shape) != (p.dtype, p.shape):
            raise TypeError(
                f"jvp: a tangent must have its primal's type, got {t.dtype}{list(t.shape)} for {p.dtype}{list(p.shape)}"
            )

        if _is_float(p.dtype):
            tangent[p.var] = t.var

    _jvp_tape(tape, tangent)

    return out, tracer._tree_map(lambda o: _value(tangent[o.var]) if o.var in tangent else _full(o, 0), out)


def linearize(fn, *primals):
    """``fn(*primals)``, and its linear program at them: ``(seeds,
    linear, tangents)``, the tangent seed of each primal (by var: a
    placeholder), the linear nodes (dependent on the seeds) in order, and
    the tangent of each output (by var). JAX's ``ad.linearize``."""
    leaves = _check("linearize", primals)
    out, tape = _record(fn, *primals)
    return out, _linearize_tape(tape, leaves)


def _linearize_tape(tape, leaves):
    """The linear program of the nodes ``tape`` recorded, at ``leaves``:
    ``(seeds, linear, linear_vars, tangents)``, as :func:`linearize`. Each
    attention among them first becomes one node, whose backward is flash
    attention's (``attention.substitute``); dots sharing an operand, one
    node differentiated as the one dot they run as (``dots.substitute``)."""
    tape = dots.substitute(attention.substitute(tape), leaves)
    seeds = {p.var: _full(p, 0).var for p in leaves if _is_float(p.dtype)}
    tangents = dict(seeds)
    _, lin_tape = _record(_jvp_tape, tape, tangents, True)
    # Partial evaluation: the nodes depending on a seed are linear.
    linear_vars = set(seeds.values())
    linear = []
    for node in lin_tape:
        if any(v in linear_vars for v in node[1]):
            linear_vars.update(node[3] if node[0] == _FUNCTION else [node[3]])
            linear.append(node)
    return seeds, linear, linear_vars, tangents


def backward(out, ct, tape, leaves):
    """The gradient of each of ``leaves`` (None for one ``out`` does not
    depend on), ``out``'s cotangent ``ct``, through the nodes ``tape``
    recorded (``TracedTensor.backward``)."""

    if (ct.dtype, ct.shape) != (out.dtype, out.shape):
        raise TypeError(f"backward: a gradient must have its value's type, got {ct.dtype}{list(ct.shape)} for {out!r}")

    seeds, linear, linear_vars, tangents = _linearize_tape(tape, leaves)
    t = tangents.get(out.var)
    cts = backward_pass(linear, linear_vars, {t: ct.var} if t in linear_vars else {})

    return [_value(cts[seeds[p.var]]) if seeds.get(p.var) in cts else None for p in leaves]


def backward_pass(linear, linear_vars, cts):
    """The cotangents of the linear program ``linear`` (nodes in order):
    ``cts`` (var -> cotangent) of its outputs, extended to its inputs',
    each node's transpose rule applied in reverse order (JAX's
    ``ad.backward_pass``)."""
    graph = tracer.current_graph()
    for name, inputs, params, out in reversed(linear):
        if name == _FUNCTION:
            _function_transpose(params, out, cts)
            continue
        ct = cts.pop(out, None)
        if ct is None:
            continue
        operands = [
            UndefinedPrimal(graph.type_of(v)[1], graph.type_of(v)[0]) if v in linear_vars else _value(v)
            for v in inputs
        ]
        primitive = getattr(prims, name, None)
        if primitive not in primitive_transposes:
            raise NotImplementedError(f"{name} has no transpose rule")
        for v, c in zip(inputs, primitive_transposes[primitive](_value(ct), *operands, **params)):
            if c is not None and v in linear_vars:
                _accumulate(cts, v, c)
    return cts


def vjp(fn, *primals):
    """``fn(*primals)``, and a function from a cotangent of it (of its
    structure) to those of ``primals`` (a tuple, each of its primal's
    structure): reverse mode, ``jax.vjp``."""
    out, (seeds, linear, linear_vars, tangents) = linearize(fn, *primals)

    def vjp_fn(ct):
        cts = {}
        for o, c in zip(tracer._tree_leaves(out), tracer._tree_leaves(ct)):
            if (c.dtype, c.shape) != (o.dtype, o.shape):
                raise TypeError(
                    f"vjp: a cotangent must have its value's type, got {c.dtype}{list(c.shape)} for {o.dtype}{list(o.shape)}"
                )
            t = tangents.get(o.var)
            if t in linear_vars:
                cts[t] = prims.add(_value(cts[t]), c).var if t in cts else c.var
        cts = backward_pass(linear, linear_vars, cts)

        def cotangent(p):
            seed = seeds.get(p.var)
            return _value(cts[seed]) if seed in cts else _full(p, 0)

        return tuple(tracer._tree_map(cotangent, p) for p in primals)

    return out, vjp_fn


def value_and_grad(fn, argnums=0):
    """A function returning ``fn``'s value (a float scalar) and its
    gradient with respect to argument ``argnums`` (a tensor, or a module:
    a module of its gradients), or each of a tuple of them: ``jax.value_and_grad``.
    Like every op, it runs while tracing (inside ``lumen.compile``)."""

    def value_and_grad_fn(*args):
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)

        def f(*diff):
            full = list(args)
            for i, d in zip(nums, diff):
                full[i] = d
            return fn(*full)

        out, pullback = vjp(f, *(args[i] for i in nums))
        if not isinstance(out, TracedTensor) or out.shape != () or not _is_float(out.dtype):
            raise TypeError(f"grad: fn must return a float scalar, got {out!r}")
        grads = pullback(_full(out, 1))
        return out, grads[0] if isinstance(argnums, int) else grads

    return value_and_grad_fn


def grad(fn, argnums=0):
    """The gradient of ``fn`` (returning a float scalar) with respect to
    argument ``argnums``, as :func:`value_and_grad` (``jax.grad``)."""
    value_and_grad_fn = value_and_grad(fn, argnums)

    @functools.wraps(fn)
    def grad_fn(*args):
        return value_and_grad_fn(*args)[1]

    return grad_fn
