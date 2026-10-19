"""Elementwise primitives on TPU, in lumen's Pallas: nodes of one shape's
operands over the device's memory.

``lumen/ops/elementwise/tpu/mod.rs`` writes the spec (the operands' and output's
dtypes, their shape, the nodes and the blocks); this is the kernel. The
operands and output are viewed in place, in a TPU's tiles: as
``[lead, m, n]`` (their leading dimensions merged) or, of one dimension or
none, as ``[elements]``. One pipeline (``emit_pipeline``, its copies
double-buffered against compute) runs over their blocks: each step loads
every operand's block, applies the nodes (lumen's primitives, which the
kernel lowers as Pallas does) and stores the last one's value as the
output's block. A 0-d operand (a broadcast's) is copied to VMEM once,
before the pipeline, and read as a scalar at every step; a ``full`` or a
broadcast computes the blocks' shape. A 0-d program has no pipeline: its
operands copied in, its value computed and copied out.
"""

import math

from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def program(spec):
    """The kernel of ``spec``: ``shape``, ``inputs`` (the operands' dtypes),
    ``broadcast`` (whether each operand is 0-d, the output not), ``nodes``
    (each a primitive's ``name``, ``params`` and ``args``: indices of the
    values, the operands then the nodes before), ``output`` (its dtype) and
    ``blocks`` (of the last two dimensions, or of the elements).
    Its scalars are the operands' rows, then the output's."""
    shape, dtypes, broadcast = spec["shape"], spec["inputs"], spec["broadcast"]
    if len(shape) >= 2:
        (bm, bn), (m, n) = spec["blocks"], shape[-2:]
        view, block_shape = (math.prod(shape[:-2]), m, n), (bm, bn)
        block = pl.BlockSpec((None, bm, bn), lambda l, i, j: (l, i, j))
        grid = (view[0], m // bm, n // bn)
    else:
        (b,) = spec["blocks"]
        view, block_shape = (max(math.prod(shape), 1),), (b,)
        block = pl.BlockSpec((b,), lambda i: (i,))
        grid = (view[0] // b,)

    scalar = not shape

    def kernel(rows, memory, result):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        def staged(dtype):
            # As `pallas_call` stages a scalar: a (1,) in VMEM (the same
            # bytes), read as a scalar.
            vmem = pl.scratch(pl.VMEM((1,), dtype))
            return vmem, type(vmem)(vmem.kernel, vmem.var, shape=())

        piped, scalars = [], {}
        for i, dtype in enumerate(dtypes):
            if not (broadcast[i] or scalar):
                piped.append(pallas.view(memory, rows[i], view, dtype))
                continue
            vmem, scalars[i] = staged(dtype)
            copy(pallas.view(memory, rows[i], (), dtype), vmem)

        def body(*blocks):
            blocks = iter(blocks)
            values = [scalars[i][...] if i in scalars else next(blocks)[...] for i in range(len(dtypes))]
            for node in spec["nodes"]:
                args = [values[i] for i in node["args"]]
                params = node["params"]
                if node["name"] in ("full", "broadcast_in_dim"):
                    params = {**params, "shape": () if scalar else block_shape}
                values.append(prims.bind(node["name"], *args, **params))
            next(blocks)[...] = values[-1]

        if scalar:
            vmem, value = staged(spec["output"])
            body(value)
            copy(vmem, pallas.view(result, rows[len(dtypes)], (), spec["output"]))
            return
        out = pallas.view(result, rows[len(dtypes)], view, spec["output"])
        pl.emit_pipeline(body, grid=grid, in_specs=[block] * len(piped), out_specs=[block])(*piped, out)

    return pallas.program(kernel, spec)
