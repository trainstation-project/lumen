"""Merges dots that share an lhs and differ only in their weight's free dimension, for one backward."""

from lumen.autograd.core import UndefinedPrimal, _value
from lumen.autograd.function import _FUNCTION, Function, FunctionCtx
from lumen.autograd.rules import _dot_dims, _dot_general_transpose
from lumen.graph import prims, tracer

__all__ = ["substitute"]


def _free(rank, *used):
    """The dimensions of a dot operand of ``rank`` in none of ``used``."""
    return [d for d in range(rank) if all(d not in u for u in used)]


def substitute(tape, leaves):
    """``tape`` with each group of dots sharing their lhs, their rhs distinct
    float weights among ``leaves`` alike but in their one free dimension,
    replaced by one ``_MergedDot`` node (where the first was)."""
    if not tracer.compiler_config().merge_dots:
        return tape
    graph = tracer.current_graph()
    weights = {t.var for t in leaves if not t.weak}
    groups = {}
    for k, (name, inputs, params, out) in enumerate(tape):
        if name != "dot_general" or inputs[1] not in weights or inputs[0] == inputs[1]:
            continue
        (_, rc), (_, rb) = _dot_dims(params)
        dtype, shape = graph.type_of(inputs[1])
        free = _free(len(shape), rc, rb)
        if len(free) != 1:
            continue
        # Group by lhs, dot params, weight dtype and weight shape with the free dimension wildcarded.
        rest = tuple(n if d != free[0] else None for d, n in enumerate(shape))
        key = (inputs[0], tuple(sorted(params.items())), dtype, rest)
        groups.setdefault(key, []).append(k)
    for ks in groups.values():
        # Keep only the first dot of each distinct weight.
        ks = [k for i, k in enumerate(ks) if tape[k][1][1] not in {tape[j][1][1] for j in ks[:i]}]
        if len(ks) < 2:
            continue
        x = _value(tape[ks[0]][1][0])
        ws = [_value(tape[k][1][1]) for k in ks]
        ctx = FunctionCtx()
        ctx.params, ctx.x, ctx.ws = tape[ks[0]][2], x, ws
        node = (
            _FUNCTION,
            [x.var] + [w.var for w in ws],
            {"function": _MergedDot, "ctx": ctx, "entries": list(enumerate([x, *ws])), "nargs": 1 + len(ws)},
            tuple(tape[k][3] for k in ks),
        )
        tape = [node if i == ks[0] else e for i, e in enumerate(tape) if i not in ks[1:]]
        # Restart so tape indices of the remaining groups stay valid.
        return substitute(tape, leaves)
    return tape


class _MergedDot(Function):
    """Dots ``x @ w_i`` (as :func:`substitute` found them; never applied:
    their forward is the program's), differentiated as the one dot
    ``x @ [w_1 | ...]``."""

    @staticmethod
    def backward(ctx, *grads):
        """Gradients of ``x`` and each ``w_i``, from one transposed dot over the concatenated weights."""
        p, x, ws = ctx.params, ctx.x, ctx.ws
        (_, rc), (_, rb) = _dot_dims(p)
        (dim,) = _free(ws[0].ndim, rc, rb)
        w = prims.concatenate(ws, dim)
        g = prims.concatenate(list(grads), grads[0].ndim - 1)
        dx = _dot_general_transpose(g, UndefinedPrimal(x.shape, x.dtype), w, **p)[0]
        dw = _dot_general_transpose(g, x, UndefinedPrimal(w.shape, w.dtype), **p)[1]
        out, start = [dx], 0
        # Split the merged weight gradient back into each weight's slice.
        for wi in ws:
            starts, limits = [0] * dw.ndim, list(dw.shape)
            starts[dim], limits[dim] = start, start + wi.shape[dim]
            out.append(prims.slice(dw, starts, limits))
            start += wi.shape[dim]
        return tuple(out)
