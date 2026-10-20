"""Dots sharing an operand, differentiated as one: the pass before autodiff
that makes the dots ``x @ w1``, ``x @ w2``, ... (one ``x``, weights side by
side: a block the MPS compiler packs and runs them as one dot of, XLA's
``DotMerger``) one node of the tape, the derivative of the dot that runs,
``x @ [w1 | w2 | ...]``: its backward one dot for ``x``'s gradient,
``[g1 | g2 | ...] @ [w1 | w2 | ...]^T``, and one for the weights',
``x^T @ [g1 | g2 | ...]``, sliced into each's. The cotangents side by side
are a ``concatenate`` the compiler fuses into the kernels writing them; the
weights side by side, the block they are placed in.

As the MPS compiler merges (``graph_compiler/mps/merge_dots.rs``): the dots
whose results an elementwise value combines (K of them: SwiGLU's gate and
up) one group if K divides a SIMD block's 8 columns, else left apart; the
others, each read on its own, one group.

With ``lumen.config.compiler.merge_dots`` off, nothing is merged.
"""

from lumen.autograd.core import UndefinedPrimal, _value
from lumen.autograd.function import _FUNCTION, Function, FunctionCtx
from lumen.autograd.rules import _dot_dims, _dot_general_transpose
from lumen.graph import prims, tracer
from lumen.graph.tracer import config

__all__ = ["substitute"]


def _free(rank, *used):
    """The dimensions of a dot operand of ``rank`` in none of ``used``."""
    return [d for d in range(rank) if all(d not in u for u in used)]


def substitute(tape, leaves):
    """``tape`` with each group of dots sharing their lhs, their rhs distinct
    float weights among ``leaves`` alike but in their one free dimension,
    replaced by one ``_MergedDot`` node (where the first was)."""
    if not config.compiler.merge_dots:
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
        rest = tuple(n if d != free[0] else None for d, n in enumerate(shape))
        key = (inputs[0], tuple(sorted(params.items())), dtype, rest)
        groups.setdefault(key, []).append(k)
    for ks in (part for ks in groups.values() for part in _split_by_epilogue(graph, tape, ks)):
        # Distinct weights, two or more.
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
        # Positions after a removed entry moved: regroup on the new tape.
        return substitute(tape, leaves)
    return tape


# The columns of a SIMD group's accumulator block, as merge_dots.rs's
# `SIMD_BLOCK`: K combined values in one block only if K divides it.
_SIMD_BLOCK = 8


def _split_by_epilogue(graph, tape, ks):
    """The dots at tape positions ``ks`` (sharing an operand) split as
    ``merge_dots.rs``'s ``split_by_epilogue``: each set an elementwise
    value combines (K of them) its own if K divides ``_SIMD_BLOCK``, else
    none; the others, each read on its own, together."""
    sets = graph.combined([tape[k][3] for k in ks])
    members = {}
    for k, s in zip(ks, sets):
        members.setdefault(s, []).append(k)
    alone = [m[0] for m in members.values() if len(m) == 1]
    parts = [m for m in members.values() if len(m) > 1 and _SIMD_BLOCK % len(m) == 0]
    return parts + ([alone] if len(alone) > 1 else [])


class _MergedDot(Function):
    """Dots ``x @ w_i`` (as :func:`substitute` found them; never applied:
    their forward is the program's), differentiated as the one dot
    ``x @ [w_1 | ...]``."""

    @staticmethod
    def backward(ctx, *grads):
        p, x, ws = ctx.params, ctx.x, ctx.ws
        (_, rc), (_, rb) = _dot_dims(p)
        (dim,) = _free(ws[0].ndim, rc, rb)
        # The weights and the cotangents side by side: each weight's free
        # dimension, each result's last.
        w = prims.concatenate(ws, dim)
        g = prims.concatenate(list(grads), grads[0].ndim - 1)
        dx = _dot_general_transpose(g, UndefinedPrimal(x.shape, x.dtype), w, **p)[0]
        dw = _dot_general_transpose(g, x, UndefinedPrimal(w.shape, w.dtype), **p)[1]
        out, start = [dx], 0
        for wi in ws:
            starts, limits = [0] * dw.ndim, list(dw.shape)
            starts[dim], limits[dim] = start, start + wi.shape[dim]
            out.append(prims.slice(dw, starts, limits))
            start += wi.shape[dim]
        return tuple(out)
