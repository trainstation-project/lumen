from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def program(spec):
    """The sum of ``spec``: ``rank``, ``lead``, ``h``, ``m``, ``n``, the
    ``input`` and ``output`` dtypes and ``blocks`` (bh, bm, bn). Its scalars
    are the rows of the operand and the output."""
    lead, h, m, n = spec["lead"], spec["h"], spec["m"], spec["n"]
    bh, bm, bn = spec["blocks"]
    dtype, out_dtype = spec["input"], spec["output"]
    nb = n // bn
    if spec["rank"] == 2:
        views, axis = ((m, n), (m,)), 1
        x_spec = pl.BlockSpec((bm, bn), lambda i, s: (i, s))
        o_spec, o_block = pl.BlockSpec((bm,), lambda i, s: (i,)), (bm,)
        grid = (m // bm, nb)
    else:
        views, axis = ((lead, h, m, n), (lead, h, m)), 2
        x_spec = pl.BlockSpec((None, bh, bm, bn), lambda l, i, j, s: (l, i, j, s))
        o_spec, o_block = pl.BlockSpec((None, bh, bm), lambda l, i, j, s: (l, i, j)), (bh, bm)
        grid = (lead, h // bh, m // bm, nb)

    def partial(xb):
        v = xb[...]
        if dtype != out_dtype:
            v = prims.cast(v, out_dtype)
        return prims.reduce_sum(v, (axis,), out_dtype)

    def kernel(rows, memory, result, *scratch):
        x = pallas.view(memory, rows[0], views[0], dtype)
        out = pallas.view(result, rows[1], views[1], out_dtype)

        def body(xb, ob):
            if nb == 1:
                ob[...] = partial(xb)
                return
            step, acc = pl.program_id(len(grid) - 1), scratch[0]

            @pl.when(step == 0)
            def _():
                acc[...] = 0

            acc[...] += partial(xb)

            @pl.when(step == nb - 1)
            def _():
                ob[...] = acc[...]

        pl.emit_pipeline(body, grid=grid, in_specs=[x_spec], out_specs=[o_spec])(x, out)

    return pallas.program(kernel, spec, scratch=[pl.VMEM(o_block, out_dtype)] if nb > 1 else [])
