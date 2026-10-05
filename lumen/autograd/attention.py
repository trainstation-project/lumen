"""Attention's derivative, FlashAttention-2's: the pass before autodiff that
finds the attentions a traced program computes (however written: the
compiler's matcher, ``lumen/compiler/attention.rs``) and makes each one node
of the tape, its backward recomputing the probabilities from each query's
log-sum-exp rather than reading them (MLX's ``ScaledDotProductAttentionVJP``).

The MPS compiler then runs the forward as one flash-attention kernel
(writing the log-sum-exp too) and the backward as two (dK and dV, dQ). With
``lumen.config.compiler.flash_attention`` off, nothing is replaced: the
gradient is autodiff of the primitives as traced.
"""

from lumen.autograd.core import _value
from lumen.autograd.function import _FUNCTION, Function, FunctionCtx
from lumen.graph import prims, tracer
from lumen.graph.tracer import config

__all__ = ["substitute"]


def substitute(tape):
    """``tape`` with each attention whose nodes it recorded replaced by one
    node, an ``_Attention`` of the dots' operands: its output the second
    dot's, its log-sum-exp (``max + log(sum)``, from the softmax's own max
    and sum) added to the graph for its backward."""
    if not config.compiler.flash_attention:
        return tape
    graph = tracer.current_graph()
    for m in graph.attentions():
        at = {entry[3]: k for k, entry in enumerate(tape) if not isinstance(entry[3], tuple)}
        if not all(v in at for v in m["members"]):
            continue
        members = set(m["members"])
        ctx = FunctionCtx()
        ctx.dot1, ctx.dot2 = tape[at[m["scores"]]][2], tape[at[m["out"]]][2]
        ctx.chain, ctx.causal, ctx.dropout = m["chain"], m["causal"], m["dropout"]
        q, k, v, o = (_value(m[name]) for name in ("q", "k", "v", "out"))
        lse = prims.add(_value(m["max"]), prims.log(_value(m["sum"])))
        ctx.save_for_backward(q, k, v, o, lse=lse)
        node = (
            _FUNCTION,
            [q.var, k.var, v.var],
            {"function": _Attention, "ctx": ctx, "entries": [(0, q), (1, k), (2, v)], "nargs": 3},
            (o.var,),
        )
        tape = [node if e[3] == o.var else e for e in tape if e[3] not in members or e[3] == o.var]
    return tape


def _free(rank, *used):
    """The one dimension of a dot operand of ``rank`` in none of ``used``."""
    return next(d for d in range(rank) if all(d not in u for u in used))


class _Attention(Function):
    """An attention found by :func:`substitute` (never applied: its forward
    is the program's): ``o = softmax(chain(q' k'^T)) v'`` over the dots'
    batch dimensions, as the compiler matched it."""

    @staticmethod
    def backward(ctx, do):
        # FlashAttention-2's, on the scores transposed, S^T = K Q^T
        # [batch..., Sk, Sq] (a row a key, as dV = P^T dO and dK = dS^T Q
        # read them): not the forward's Q K^T, so recomputed, never its
        # [Sq, Sk] values read back.
        from lumen import functional as F  # it imports this module's package

        q, k, v, o = ctx.saved_tensors
        lse = ctx.saved_named_tensors["lse"]
        p1, p2 = ctx.dot1, ctx.dot2
        lc1, rc1, lb1, rb1 = p1["lhs_contracting"], p1["rhs_contracting"], p1["lhs_batch"], p1["rhs_batch"]
        rc2, rb2 = p2["rhs_contracting"], p2["rhs_batch"]
        nb, accum, dtype = len(lb1), p1["accum_dtype"], q.dtype
        qf, kf, vf = _free(q.ndim, lb1, lc1), _free(k.ndim, rb1, rc1), _free(v.ndim, rb2, rc2)
        batch, sq, sk = list(o.shape[:nb]), o.shape[nb], k.shape[kf]
        every = tuple(range(nb))
        # S^T, through the forward's roundings and scale; its causal mask.
        st = prims.dot_general(k, q, ((rc1, lc1), (rb1, lb1)), accum, p1["output_dtype"])
        scale = 1.0
        for step in ctx.chain[1:]:
            if step[0] == "round":
                st = prims.cast(st, step[1])
            else:
                st = prims.mul(st, prims.full(st.shape, step[1], step[2]))
                scale *= step[1]
        if ctx.causal is not None:
            keys = prims.iota("int64", st.shape, nb)
            queries = prims.iota("int64", st.shape, nb + 1)
            st = F.where(F.le(keys, queries + ctx.causal), st, float("-inf"))
        # P^T, from each query's log-sum-exp; D = rowsum(dO * O), per query.
        per_query = list(range(nb + 2))
        lse = prims.broadcast_in_dim(prims.reshape(lse, batch + [1, sq]), batch + [sk, sq], per_query)
        pt = prims.exp(prims.sub(st, lse))
        g = do if do.dtype == dtype else prims.cast(do, dtype)
        # A dot over the value head, accumulated in float32 (dO has O's
        # dtype): no rounded value widened.
        rows = tuple(range(nb + 1))
        d = prims.dot_general(do, o, (((nb + 1,), (nb + 1,)), (rows, rows)), accum, accum)
        d = prims.broadcast_in_dim(prims.reshape(d, batch + [1, sq]), batch + [sk, sq], per_query)
        # dP^T = V dO^T (dropped, as the forward's probabilities); dS^T =
        # P^T (dP^T - D) * scale.
        dpt = prims.dot_general(v, g, (((vf,), (nb + 1,)), (rb2, every)), accum, accum)
        if ctx.dropout is not None:
            swap = list(range(nb)) + [nb + 1, nb]
            dropped, keep = prims.transpose(_value(ctx.dropout["mask"]), swap), ctx.dropout["scale"]
            dpt = F.where(dropped, 0.0, dpt * keep)
        ds = prims.mul(prims.mul(pt, prims.sub(dpt, d)), prims.full(pt.shape, scale, accum))
        # dV reads P^T dropped.
        if ctx.dropout is not None:
            pt = F.where(dropped, 0.0, pt * keep)
        ds, pt = (t if dtype == accum else prims.cast(t, dtype) for t in (ds, pt))
        # [batch..., Sk, Hv], [batch..., Sk, H], [batch..., Sq, H], in float32.
        dv = prims.dot_general(pt, g, (((nb + 1,), (nb,)), (every, every)), accum, accum)
        dk = prims.dot_general(ds, q, (((nb + 1,), (qf,)), (every, lb1)), accum, accum)
        dq = prims.dot_general(ds, k, (((nb,), (kf,)), (every, rb1)), accum, accum)

        def operand(grad, x, batch_dims, row, col):
            # Back to the operand's dimensions, in its dtype.
            order = {d: i for i, d in enumerate(batch_dims)} | {row: nb, col: nb + 1}
            perm = [order[d] for d in range(x.ndim)]
            grad = grad if perm == list(range(x.ndim)) else prims.transpose(grad, perm)
            return grad if grad.dtype == x.dtype else prims.cast(grad, x.dtype)

        return (
            operand(dq, q, lb1, qf, lc1[0]),
            operand(dk, k, rb1, kf, rc1[0]),
            operand(dv, v, rb2, rc2[0], vf),
        )
