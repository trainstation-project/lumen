"""Elementwise primitives on TPU, in lumen's Pallas: nodes of one shape's
operands over the device's memory.

``lumen/ops/elementwise/tpu.rs`` writes the spec (the operands' and output's
dtypes, their shape, the nodes and the blocks); this is the kernel. The
operands and output are viewed in place, in a TPU's tiles: as
``[lead, m, n]`` (their leading dimensions merged) or, of one dimension or
none, as ``[elements]``. One pipeline (``emit_pipeline``, its copies
double-buffered against compute) runs over their blocks: each step loads
every operand's block, applies the nodes (lumen's primitives, which the
kernel lowers as Pallas does) and stores the last one's value as the
output's block.
"""

import math

from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def program(spec):
    """The kernel of ``spec``: ``shape``, ``inputs`` (the operands' dtypes),
    ``nodes`` (each a primitive's ``name``, ``params`` and ``args``: indices
    of the values, the operands then the nodes before), ``output`` (its
    dtype) and ``blocks`` (of the last two dimensions, or of the elements).
    Its scalars are the operands' rows, then the output's."""
    shape, dtypes = spec["shape"], spec["inputs"]
    if len(shape) >= 2:
        (bm, bn), (m, n) = spec["blocks"], shape[-2:]
        view = (math.prod(shape[:-2]), m, n)
        block = pl.BlockSpec((None, bm, bn), lambda l, i, j: (l, i, j))
        grid = (view[0], m // bm, n // bn)
    else:
        (b,) = spec["blocks"]
        view = (max(math.prod(shape), 1),)
        block = pl.BlockSpec((b,), lambda i: (i,))
        grid = (view[0] // b,)

    def kernel(rows, memory, result):
        operands = [pallas.view(memory, rows[i], view, dtype) for i, dtype in enumerate(dtypes)]
        out = pallas.view(result, rows[len(dtypes)], view, spec["output"])

        def body(*blocks):
            values = [b[...] for b in blocks[:-1]]
            for node in spec["nodes"]:
                args = [values[i] for i in node["args"]]
                values.append(prims.bind(node["name"], *args, **node["params"]))
            blocks[-1][...] = values[-1]

        pl.emit_pipeline(body, grid=grid, in_specs=[block] * len(dtypes), out_specs=[block])(*operands, out)

    return pallas.program(kernel, spec)
