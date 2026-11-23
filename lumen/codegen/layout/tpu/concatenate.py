"""Pallas concatenation for TPU: copies each operand of a TPU-tiled concatenation into its slot of the result."""

import math

import lumen.functional as F
from lumen.codegen.devices.tpu import pallas
from lumen.kernels import pallas as pl


def program(spec):
    """The concatenation of ``spec``: operands ``sizes`` long along
    ``dimension`` of the result ``shape``, each its step input
    ``operands[k]`` (of ``inputs``; one value may be several operands). ``kind`` ``dma`` copies each
    operand into its slot (every slot starts on a tile); ``lanes`` (along the
    last axis) and ``rows`` (along the one before) build each ``block``-row
    or ``block``-column block of the result from its operands' blocks; an
    empty operand is skipped. Its scalars are the operands' rows, then the
    result's."""
    shape, d, sizes, kind, block, dtype = (spec[k] for k in ("shape", "dimension", "sizes", "kind", "block", "dtype"))
    rank, count = len(shape), len(sizes)
    operands_of, inputs = spec["operands"], spec["inputs"]
    starts = [sum(sizes[:k]) for k in range(count)]
    lead = math.prod(shape[:-2])
    m, n = shape[-2:]

    def slot(size, start):
        """The 3- or 4-d view of an operand ``size`` long and its slot's index in the result's view."""
        if d < rank - 2:
            # Leading axes: slabs `before` x (`size` x `after`) of whole (m, n) matrices.
            before, after = math.prod(shape[:d]), math.prod(shape[d + 1 : -2])
            return (before, size * after, m, n), (slice(None), pl.ds(start * after, size * after))
        if d == rank - 2:
            return (lead, size, n), (slice(None), pl.ds(start, size))
        return (lead, m, size), (slice(None), slice(None), pl.ds(start, size))

    def kernel(rows, memory, result):
        """Copies each operand into its slot of the result."""
        out_shape = slot(shape[d], 0)[0]
        out = pallas.view(result, rows[inputs], out_shape, dtype)
        # Each non-empty operand: its view, size and start.
        operands = [
            (pallas.view(memory, rows[operands_of[k]], slot(size, 0)[0], dtype), size, start)
            for k, (size, start) in enumerate(zip(sizes, starts))
            if size
        ]

        if kind == "dma":
            sems = pl.scratch(pl.SemaphoreType.DMA((len(operands),)))
            copies = [
                pl.make_async_copy(x, out.at[slot(size, start)[1]], sems.at[k])
                for k, (x, size, start) in enumerate(operands)
            ]
            for c in copies:
                c.start()
            for c in copies:
                c.wait()
            return

        def body(*refs):
            """One block of the result, its operands' blocks side by side."""
            *xbs, ob = refs
            ob[...] = F.cat([xb[...] for xb in xbs], -1 if kind == "lanes" else 0)

        if kind == "lanes":
            grid = (lead, m // block)
            in_specs = [pl.BlockSpec((None, block, size), lambda l, i: (l, i, 0)) for _, size, _ in operands]
            out_spec = pl.BlockSpec((None, block, n), lambda l, i: (l, i, 0))
        else:
            grid = (lead, n // block)
            in_specs = [pl.BlockSpec((None, size, block), lambda l, j: (l, 0, j)) for _, size, _ in operands]
            out_spec = pl.BlockSpec((None, m, block), lambda l, j: (l, 0, j))

        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=[out_spec])(*(x for x, _, _ in operands), out)

    return pallas.program(kernel, spec)
