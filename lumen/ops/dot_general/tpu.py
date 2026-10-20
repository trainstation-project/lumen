from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def program(spec, epilogue=None):
    """The dot of ``spec``: ``batch``, ``lead``, ``m``, ``n``, ``k``,
    ``transposed`` (a ``[n, k]`` rhs), the ``operands`` and ``output``
    dtypes, ``blocks`` (bm, bn, bk) and ``outputs`` (the widths of the
    column ranges written to outputs of their own, in order; none: one
    output). Its scalars are the rows of the lhs, the rhs and each output.
    ``epilogue`` (float32 block to float32 block,
    elementwise) is applied to each output block before it is rounded."""
    batch, lead, m, n, k = spec["batch"], spec["lead"], spec["m"], spec["n"], spec["k"]
    bm, bn, bk = spec["blocks"]
    t = spec["transposed"]
    dtype, out_dtype = spec["operands"], spec["output"]
    mb, nb, kb = m // bm, n // bn, k // bk
    widths = spec.get("outputs") or [n]
    starts = [sum(widths[:p]) // bn for p in range(len(widths))]
    counts = [w // bn for w in widths]

    def rhs_batch(b):
        if lead == 1:
            return b
        if isinstance(b, int):
            return b // lead
        return prims.div(b, prims.full((), lead, "int32"))

    def kernel(rows, memory, result, *scratch):
        x = pallas.view(memory, rows[0], (batch * lead, m, k), dtype)
        y = pallas.view(memory, rows[1], (batch, n, k) if t else (batch, k, n), dtype)
        outs = [pallas.view(result, rows[2 + p], (batch * lead, m, w), out_dtype) for p, w in enumerate(widths)]
        x_spec = pl.BlockSpec((None, bm, bk), lambda b, i, j, s: (b, i, s))
        if t:
            y_spec = pl.BlockSpec((None, bn, bk), lambda b, i, j, s: (rhs_batch(b), j, s))
        else:
            y_spec = pl.BlockSpec((None, bk, bn), lambda b, i, j, s: (rhs_batch(b), s, j))

        def column(start, count):
            def index(b, i, j, s):
                if isinstance(j, int):
                    return b, i, min(max(j - start, 0), count - 1)
                shifted = prims.max(prims.sub(j, prims.full((), start, "int32")), prims.full((), 0, "int32"))
                return b, i, prims.neg(prims.max(prims.neg(shifted), prims.full((), 1 - count, "int32")))

            return pl.BlockSpec((None, bm, bn), index)

        o_specs = [column(start, count) for start, count in zip(starts, counts)]

        def finish(v, obs):
            if epilogue is not None:
                v = epilogue(v)
            v = v.to(out_dtype)
            if len(obs) == 1:
                obs[0][...] = v
                return
            j = pl.program_id(2)
            for ob, start, count in zip(obs, starts, counts):

                @pl.when((j >= start) & (j < start + count))
                def _(ob=ob):
                    ob[...] = v

        def body(xb, yb, *obs):
            if kb == 1:
                finish(pl.dot(xb[...], yb[...], trans_b=t), obs)
                return
            step, acc = pl.program_id(3), scratch[0]

            @pl.when(step == 0)
            def _():
                acc[...] = 0.0

            acc[...] += pl.dot(xb[...], yb[...], trans_b=t)

            @pl.when(step == kb - 1)
            def _():
                finish(acc[...], obs)

        grid = (batch * lead, mb, nb, kb)
        pl.emit_pipeline(body, grid=grid, in_specs=[x_spec, y_spec], out_specs=o_specs)(x, y, *outs)

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bm, bn), "float32")] if kb > 1 else [])
