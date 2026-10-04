"""Custom derivatives: ``torch.autograd.Function`` (JAX: ``custom_vjp``)."""

import inspect

from lumen.autograd.core import _accumulate, _full, _value
from lumen.graph import tracer
from lumen.graph.tracer import TracedTensor

# The tape's name for a Function's call: ``(_FUNCTION, its tensor
# arguments' vars, {"function", "ctx", "args"}, its outputs' vars)``.
_FUNCTION = "autograd.Function"


class FunctionCtx:
    def save_for_backward(self, *tensors, **named):
        self.saved_tensors = tensors
        self.saved_named_tensors = named


class Function:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("a Function defines forward(ctx, *args)")

    @staticmethod
    def backward(ctx, *grads):
        raise NotImplementedError("a Function defines backward(ctx, *grads) for reverse mode")

    @staticmethod
    def jvp(ctx, *tangents, **kw_tangents):
        raise NotImplementedError("a Function defines jvp(ctx, *tangents, **kw_tangents) for forward mode")

    @classmethod
    def apply(cls, *args, **kwargs):
        """``forward(ctx, *args, **kwargs)``, recorded as one node whose
        derivative is ``backward``.
        Arguments bind to ``forward``'s parameters (defaults included).
        ``backward`` returns the gradients of the positional ones in order
        (a tuple), of any by name (a dict: keyword-only ones too), or the
        first positional ones then a dict; None, or a name left out, is
        none.
        ``jvp`` takes their tangents so too: positional, then by name."""

        bound = inspect.signature(cls.forward).bind(None, *args, **kwargs)
        bound.apply_defaults()

        args = bound.args[1:]
        kwargs = bound.kwargs

        tracer.current_graph()
        ctx = FunctionCtx()

        tapes = tracer._TAPES[:]
        tracer._TAPES.clear()

        try:
            out = cls.forward(ctx, *args, **kwargs)
        finally:
            tracer._TAPES[:] = tapes

        outs = (out,) if isinstance(out, TracedTensor) else tuple(out)
        if not outs or not all(isinstance(o, TracedTensor) for o in outs):
            raise TypeError(f"{cls.__name__}.forward must return a tensor or a tuple of them, got {out!r}")

        # Each argument by key: its position, or its keyword's name.
        entries = list(enumerate(args)) + list(kwargs.items())
        inputs = [v.var for _, v in entries if isinstance(v, TracedTensor)]
        params = {"function": cls, "ctx": ctx, "entries": entries, "nargs": len(args)}

        for tape in tapes:
            tape.append((_FUNCTION, inputs, params, tuple(o.var for o in outs)))

        return out


def _function_jvp(params, ts, outs, tangents, linearizing):
    """A Function's tangents (into ``tangents``): placeholders, outputs of
    one linear node ``backward`` transposes, when ``linearizing``; else its
    ``jvp``'s."""

    by_arg = iter(ts)
    arg_tangents = [next(by_arg) if isinstance(v, TracedTensor) else None for _, v in params["entries"]]

    if linearizing:
        placeholders = tuple(_full(_value(o), 0).var for o in outs)
        node = dict(params, tangents=arg_tangents)
        tracer._TAPES[-1].append((_FUNCTION, [t for t in arg_tangents if t is not None], node, placeholders))
        tangents.update(zip(outs, placeholders))
    else:
        fn, n = params["function"], params["nargs"]
        ts = [_value(t) if t is not None else None for t in arg_tangents]
        names = [k for k, _ in params["entries"][n:]]

        out = fn.jvp(params["ctx"], *ts[:n], **dict(zip(names, ts[n:])))
        out = (out,) if isinstance(out, TracedTensor) else tuple(out)

        if len(out) != len(outs):
            raise ValueError(f"{fn.__name__}.jvp returned {len(out)} tangents for {len(outs)} outputs")
        for o, t in zip(outs, out):
            if t is not None:
                _check_type(fn.__name__ + ".jvp", t, o)
                tangents[o] = t.var


def _function_transpose(params, outs, cts):
    """A Function's linear node transposed: its ``backward`` on the
    cotangents of its outputs (zeros for those without)."""
    cs = [cts.pop(o, None) for o in outs]
    if all(c is None for c in cs):
        return
    fn, entries, n = params["function"], params["entries"], params["nargs"]
    grads = fn.backward(
        params["ctx"], *(_value(c) if c is not None else _full(_value(o), 0) for c, o in zip(cs, outs))
    )
    for (_, v), t, g in zip(entries, params["tangents"], _gradients(fn, grads, entries, n)):
        if t is not None and g is not None:
            _check_type(fn.__name__ + ".backward", g, v.var)
            _accumulate(cts, t, g)


def _gradients(fn, grads, entries, n):
    """What ``fn.backward`` returned (a tuple of the positional arguments'
    gradients, a dict of any by name, or the first positional ones then a
    dict), each entry's gradient (None for none)."""
    named = {}
    if isinstance(grads, dict):
        grads, named = (), grads
    elif isinstance(grads, (tuple, list)):
        grads = tuple(grads)
        if grads and isinstance(grads[-1], dict):
            grads, named = grads[:-1], grads[-1]
    else:
        grads = (grads,)
    # All the positional ones, or (followed by names) the first of them.
    if len(grads) > n or (not named and len(grads) != n):
        raise ValueError(f"{fn.__name__}.backward returned {len(grads)} gradients for {n} positional arguments")
    names = list(inspect.signature(fn.forward).parameters)[1:]
    by_key = dict(enumerate(grads))
    for name, g in named.items():
        key = names.index(name) if name in names[:n] else name
        if key not in dict(entries):
            raise ValueError(f"{fn.__name__}.backward returned a gradient for {name!r}, not an argument")
        by_key[key] = g
    return [by_key.get(k) for k, _ in entries]


def _check_type(where, t, v):
    """``t`` must have value ``v``'s type."""
    dtype, shape = tracer.current_graph().type_of(v)
    if (t.dtype, list(t.shape)) != (dtype, list(shape)):
        raise TypeError(f"{where}: returned {t.dtype}{list(t.shape)} for {dtype}{list(shape)}")
