from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def program(spec, epilogue=None):
    """The dot of ``spec``: ``batch``, ``lead``, ``m``, ``n``, ``k``,
    ``transposed`` (a ``[n, k]`` rhs), the ``operands`` and ``output``
    dtypes and ``blocks`` (bm, bn, bk). Its scalars are the rows of the lhs,
    the rhs and the output. ``epilogue`` (float32 block to float32 block,
    elementwise) is applied to each output block before it is rounded."""
    batch, lead, m, n, k = spec["batch"], spec["lead"], spec["m"], spec["n"], spec["k"]
    bm, bn, bk = spec["blocks"]
    t = spec["transposed"]
    dtype, out_dtype = spec["operands"], spec["output"]
    mb, nb, kb = m // bm, n // bn, k // bk

    def rhs_batch(b):
        if lead == 1:
            return b
        if isinstance(b, int):
            return b // lead
        return prims.div(b, prims.full((), lead, "int32"))

    def kernel(rows, memory, result, *scratch):
        x = pallas.view(memory, rows[0], (batch * lead, m, k), dtype)
        y = pallas.view(memory, rows[1], (batch, n, k) if t else (batch, k, n), dtype)
        out = pallas.view(result, rows[2], (batch * lead, m, n), out_dtype)
        x_spec = pl.BlockSpec((None, bm, bk), lambda b, i, j, s: (b, i, s))
        if t:
            y_spec = pl.BlockSpec((None, bn, bk), lambda b, i, j, s: (rhs_batch(b), j, s))
        else:
            y_spec = pl.BlockSpec((None, bk, bn), lambda b, i, j, s: (rhs_batch(b), s, j))
        o_spec = pl.BlockSpec((None, bm, bn), lambda b, i, j, s: (b, i, j))

        def finish(v, ob):
            if epilogue is not None:
                v = epilogue(v)
            ob[...] = v.to(out_dtype)

        def body(xb, yb, ob):
            if kb == 1:
                finish(pl.dot(xb[...], yb[...], trans_b=t), ob)
                return
            step, acc = pl.program_id(3), scratch[0]

            @pl.when(step == 0)
            def _():
                acc[...] = 0.0

            acc[...] += pl.dot(xb[...], yb[...], trans_b=t)

            @pl.when(step == kb - 1)
            def _():
                finish(acc[...], ob)

        grid = (batch * lead, mb, nb, kb)
        pl.emit_pipeline(body, grid=grid, in_specs=[x_spec, y_spec], out_specs=[o_spec])(x, y, out)

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bm, bn), "float32")] if kb > 1 else [])
