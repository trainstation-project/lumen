import math

from lumen import pallas as pl
from lumen.codegen.tpu import computed
from lumen.ops.tpu import pallas


def program(spec):
    """The kernel of ``spec``: ``shape``, ``inputs`` (the operands' dtypes),
    ``broadcast`` (whether each operand is 0-d, the output not), ``nodes``
    (each a primitive's ``name``, ``params`` and ``args``: indices of the
    values, the operands then the nodes before), ``result`` (the index of
    the value written), ``output`` (its dtype),
    ``blocks`` (of the leading dimensions together and the last two, or of
    the elements) and
    ``window`` (``[n, start]``: the output written at columns ``start`` on
    of a buffer ``n`` wide; none: a buffer of its own).
    Its scalars are the operands' rows, then the output's."""
    shape, dtypes, broadcast = spec["shape"], spec["inputs"], spec["broadcast"]
    if len(shape) >= 2:
        (bl, bm, bn), (m, n) = spec["blocks"], shape[-2:]
        view = (math.prod(shape[:-2]), m, n)
        block_shape = (bl, bm, bn) if bl > 1 else (bm, bn)
        block = pl.BlockSpec((bl if bl > 1 else None, bm, bn), lambda l, i, j: (l, i, j))
        grid = (view[0] // bl, m // bm, n // bn)
    else:
        (b,) = spec["blocks"]
        view, block_shape = (max(math.prod(shape), 1),), (b,)
        block = pl.BlockSpec((b,), lambda i: (i,))
        grid = (view[0] // b,)
    out_view, out_block = view, block
    if spec["window"]:
        whole, start = spec["window"]
        out_view = (view[0], m, whole)
        out_block = pl.BlockSpec((bl if bl > 1 else None, bm, bn), lambda l, i, j: (l, i, j + start // bn))

    scalar = not shape

    shape_of = (1,) if scalar else block_shape

    def apply(values):
        return computed(spec["nodes"], values, shape_of)[spec["result"]]

    def kernel(rows, memory, result):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        piped, scalars = [], {}
        for i, dtype in enumerate(dtypes):
            if not (broadcast[i] or scalar):
                piped.append(pallas.view(memory, rows[i], view, dtype))
                continue
            scalars[i] = pl.scratch(pl.VMEM((1,), dtype))
            copy(pallas.view(memory, rows[i], (), dtype), scalars[i])
        read = {i: s[...] for i, s in scalars.items()}
        out = pallas.view(result, rows[len(dtypes)], () if scalar else out_view, spec["output"])
        if scalar:
            vmem = pl.scratch(pl.VMEM((1,), spec["output"]))
            vmem[...] = apply(read[i] for i in range(len(dtypes)))
            copy(vmem, out)
            return

        def body(*blocks):
            blocks = iter(blocks)
            values = [read[i] if i in read else next(blocks)[...] for i in range(len(dtypes))]
            next(blocks)[...] = apply(values)

        pl.emit_pipeline(body, grid=grid, in_specs=[block] * len(piped), out_specs=[out_block])(*piped, out)

    return pallas.program(kernel, spec)
