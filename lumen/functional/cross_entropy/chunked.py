from lumen.autograd.core import _value
from lumen.autograd.function import Function
from lumen.functional.cross_entropy.utils import _one_hot
from lumen.functional.functions import amax, exp, log, matmul, sum
from lumen.graph import prims, tracer
from lumen.graph.tracer import _lift, _require_float


def _uniform(g):
    """The scalar ``g`` is broadcast from, if it is one (a mean's or a sum's
    gradient: every row's the same), else None."""
    graph = tracer.current_graph()
    producer = {n["output"]: n for n in graph.nodes()}
    v = g.var
    while v in producer and producer[v]["primitive"] in ("reshape", "broadcast_in_dim"):
        v = producer[v]["inputs"][0]
        if graph.type_of(v)[1] == []:
            return _value(v)
    return None


class _ChunkedLinearCrossEntropy(Function):
    """:func:`linear_cross_entropy`'s rows chunked (with
    ``lumen.config.compiler.fused_linear_cross_entropy_chunk_size``), and
    their gradients (XMA's, Liger's chunked fused linear cross entropy):
    ``chunk`` rows at a time, its logits, its losses and ``softmax -
    one_hot`` (one row kernel, rounded to ``h``'s dtype), then its rows of
    ``h``'s gradient and ``w``'s, added to in float32: computed with the
    losses, so the backward only scales them when every row's gradient is
    the same (any mean's or sum's: three matmuls in all); else (each row's
    own) ``h``'s scaled a row at a time, and ``w``'s from the logits
    recomputed a chunk at a time in the backward (four)."""

    @staticmethod
    def forward(ctx, h, w, target, *, chunk):
        (b, d), v = h.shape, w.shape[0]
        ctx.chunk = chunk
        loss, lse = prims.full((b,), 0.0, "float32"), prims.full((b,), 0.0, "float32")
        # h's and w's gradients of the losses' sum, in float32 (rounded
        # once, scaled, in the backward).
        dh, dw = prims.full((b, d), 0.0, "float32"), None
        for start in range(0, b, chunk):
            rows = slice(start, min(start + chunk, b))
            hc, tc = h[rows], target[rows]
            n = tc.shape[0]
            # The chunk's logits, its log-sum-exps and losses (each row's
            # logit at its class: element row * v + class), and their
            # gradient, rounded to h's dtype as the matmuls read it.
            x = matmul(hc, w.t(), "float32", "float32")
            m = amax(x, -1, keepdim=True)
            lc = m.reshape(n) + log(sum(exp(x - m), -1))
            at = prims.iota(tc.dtype, [n], 0) * v + tc
            loss = prims.dynamic_update_slice(loss, lc - prims.gather(x.reshape(n * v), at, 0), (start,))
            lse = prims.dynamic_update_slice(lse, lc, (start,))
            ds = (exp(x - lc.reshape(n, 1)) - _one_hot(tc, v, "float32")).to(dtype=h.dtype)
            dh = prims.dynamic_update_slice(dh, matmul(ds, w, "float32", "float32"), (start, 0))
            part = matmul(ds.t(), hc, "float32", "float32")
            dw = part if dw is None else dw + part
        ctx.save_for_backward(h, w, target, lse, dh, dw)
        return loss

    @staticmethod
    def backward(ctx, g):
        h, w, target, lse, dh, dw = ctx.saved_tensors
        # Every row's g the same (a mean, a sum): the forward's gradients,
        # scaled (the three matmuls in all).
        s = _uniform(g)
        if s is not None:
            return (dh * s).to(dtype=h.dtype), (dw * s).to(dtype=w.dtype), None
        # Else (each row's own: a weighted sum) h's scaled by each row's
        # (as it commutes with the matmul), and w's from the logits
        # recomputed a chunk at a time, transposed ([V, C]: unlike the
        # forward's, which an equal value would keep alive until here),
        # each row's softmax - one_hot scaled by its g.
        b, v = h.shape[0], w.shape[0]
        dw = None
        for start in range(0, b, ctx.chunk):
            rows = slice(start, min(start + ctx.chunk, b))
            hc = h[rows]
            xt = matmul(w, hc.t(), "float32", "float32")
            pt = exp(xt - lse[rows].reshape(1, -1))
            dst = ((pt - _one_hot(target[rows], v, "float32").t()) * g[rows].reshape(1, -1)).to(dtype=h.dtype)
            # w's gradient summed over the chunks in float32, rounded once.
            part = matmul(dst, hc, "float32", "float32")
            dw = part if dw is None else dw + part
        return (dh * g.reshape(-1, 1)).to(dtype=h.dtype), dw.to(dtype=w.dtype), None
