from lumen.autograd.core import _value
from lumen.autograd.function import _FUNCTION, Function, FunctionCtx
from lumen.graph import prims
from lumen.graph.tracer import config

__all__ = ["substitute"]


def substitute(tape):
    """``tape`` with each cross entropy of a dot's logits nothing else reads
    replaced by one ``_LinearCrossEntropy`` node of the dot's operands and
    the classes (where the cross entropy was), the dot dropped."""
    chunk = config.compiler.fused_linear_cross_entropy_chunk_size
    if chunk is None:
        return tape
    from lumen.functional.cross_entropy.logits import _CrossEntropy

    at = {entry[3]: k for k, entry in enumerate(tape) if not isinstance(entry[3], tuple)}
    readers = {}
    for k, (_, inputs, _, _) in enumerate(tape):
        for v in inputs:
            readers.setdefault(v, set()).add(k)
    replaced = {}
    for k, (name, inputs, params, out) in enumerate(tape):
        if name != _FUNCTION or params["function"] is not _CrossEntropy:
            continue
        d = at.get(inputs[0])
        if d is None or tape[d][0] != "dot_general" or readers[inputs[0]] != {k}:
            continue
        p = tape[d][2]
        h, w = (_value(v) for v in tape[d][1])
        rhs = tuple(p["rhs_contracting"])
        fits = (
            h.ndim == 2
            and w.ndim == 2
            and h.dtype == w.dtype
            and not p["lhs_batch"]
            and tuple(p["lhs_contracting"]) == (1,)
            and rhs in ((0,), (1,))
            and p["accum_dtype"] == "float32"
            and p["output_dtype"] == "float32"
        )
        if not fits:
            continue
        _, target, lse = params["ctx"].saved_tensors
        ctx = FunctionCtx()
        ctx.save_for_backward(h, w, target, lse)
        ctx.chunk, ctx.transposed = chunk, rhs == (0,)
        entries = [(0, h), (1, w), (2, target)]
        replaced[k] = (
            _FUNCTION,
            [h.var, w.var, target.var],
            {"function": _LinearCrossEntropy, "ctx": ctx, "entries": entries, "nargs": 3},
            out,
        )
        replaced[d] = None
    return [replaced.get(k, e) for k, e in enumerate(tape) if replaced.get(k, e) is not None]


class _LinearCrossEntropy(Function):
    """``cross_entropy(h @ w', target)`` as :func:`substitute` found it
    (never applied: its forward is the program's), differentiated a chunk of
    rows at a time from each row's log-sum-exp, the logits recomputed."""

    @staticmethod
    def backward(ctx, g):
        from lumen import functional as F
        from lumen.functional.cross_entropy.utils import _one_hot

        h, w, target, lse = ctx.saved_tensors
        wt = w if ctx.transposed else w.t()
        (b, d), v = h.shape, wt.shape[1]
        dh, dw = prims.full((b, d), 0.0, h.dtype), None
        for start in range(0, b, ctx.chunk):
            rows = slice(start, min(start + ctx.chunk, b))
            hc = h[rows]
            x = F.matmul(hc, wt, "float32", "float32")
            p = F.exp(x - lse[rows].reshape(-1, 1))
            ds = ((p - _one_hot(target[rows], v, "float32")) * g[rows].reshape(-1, 1)).to(dtype=h.dtype)
            dh = prims.dynamic_update_slice(dh, F.matmul(ds, wt.t(), "float32", h.dtype), (start, 0))
            part = F.matmul(ds.t(), hc, "float32", "float32")
            dw = part if dw is None else dw + part
        dw = dw.t() if ctx.transposed else dw
        return dh, dw.to(dtype=w.dtype), None
