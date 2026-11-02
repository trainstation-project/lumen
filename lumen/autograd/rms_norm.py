"""Rewrites RMS norms written out as primitives into one node differentiated from each row's 1 / rms."""

from lumen.autograd.core import _value
from lumen.autograd.function import _FUNCTION, FunctionCtx
from lumen.graph import prims, tracer

__all__ = ["substitute"]


def substitute(tape):
    """``tape`` with each RMS norm written out as primitives
    (``h / sqrt(mean(h * h) + eps)`` or ``h * rsqrt(..)``, as the compiler
    matches them, with the weight multiplying its rows if any) whose nodes
    it recorded replaced by one node of its operand and weight:
    ``F.rms_norm``'s, differentiated from each row's ``1 / rms`` (the
    forward's own ``rsqrt``, or its ``rms`` the backward inverts) without
    the forward's normalized value."""
    from lumen.functional.functions import _RMSNorm

    graph = tracer.current_graph()
    for m in graph.rms_norms():
        # Map each recorded output var to its tape index; tuple outputs belong to multi-output nodes.
        at = {entry[3]: k for k, entry in enumerate(tape) if not isinstance(entry[3], tuple)}
        if not all(v in at for v in m["members"]):
            continue
        members = set(m["members"])
        x, out, rms = _value(m["x"]), _value(m["out"]), _value(m["rms"])
        weight = None if m["weight"] is None else _value(m["weight"])
        # Each row's value as ``[..., 1]``, broadcasting against the rows, in the backward's accumulation dtype.
        saved = prims.reshape(rms, [*out.shape[:-1], 1])
        if saved.dtype != tracer._accum_dtype(x.dtype):
            saved = prims.cast(saved, tracer._accum_dtype(x.dtype))
        ctx = FunctionCtx()
        # An ``rms`` (not its inverse) is inverted by the backward, in its own kernel.
        ctx.reciprocal = not m["inverse"]
        ctx.save_for_backward(x, weight, saved)
        operands = [x] if weight is None else [x, weight]
        node = (
            _FUNCTION,
            [t.var for t in operands],
            {"function": _RMSNorm, "ctx": ctx, "entries": list(enumerate(operands)), "nargs": 3},
            (out.var,),
        )
        # Drop the norm's interior nodes and put the fused node where its output was recorded.
        tape = [node if e[3] == out.var else e for e in tape if e[3] not in members or e[3] == out.var]
    return tape
