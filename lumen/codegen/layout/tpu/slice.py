"""Pallas slice for TPU: a slice of a TPU-tiled tensor whose rows or columns start inside a tile."""

from lumen.codegen.devices.tpu import pallas
from lumen.kernels import pallas as pl


def program(spec):
    """The slice of ``spec``: the operand ``shape`` from ``starts``, the
    result ``out``. Each grid step makes one slab of the result (its
    leading indices): it reads the tile-aligned ``window`` (rows ``r0`` on,
    ``rw`` of them; columns ``c0`` on, ``cw`` of them) of the operand's
    slab, its leading indices shifted by the starts, and keeps the
    slice's rows and columns; a window past the operand's end is its last
    tile's padding. Its scalars are the operand's row, then the
    result's."""
    shape, out, starts, dtype = spec["shape"], spec["out"], spec["starts"], spec["dtype"]
    r0, rw, c0, cw = spec["window"]
    rank = len(out)
    lead = out[:-2]
    (r, c), (m, n) = starts[-2:], out[-2:]
    grid = tuple(lead) or (1,)

    def x_index(*l):
        """The window of grid step ``l``: element offsets, its leading indices shifted by the starts."""
        at = tuple(i + s for i, s in zip(l, starts[:-2])) if lead else ()
        return (*at, r0, c0)

    def o_index(*l):
        """The result slab of grid step ``l``."""
        return (*(l if lead else ()), 0, 0)

    # A window reaching past the operand's last row or column is padded there (its last tile's padding).
    high = [max(r0 + rw - shape[-2], 0), max(c0 + cw - shape[-1], 0)]
    window = (pl.Element(rw, (0, high[0])), pl.Element(cw, (0, high[1])))
    x_spec = pl.BlockSpec((*([None] * (rank - 2)), *window), x_index)
    o_spec = pl.BlockSpec((*([None] * (rank - 2)), m, n), o_index)

    def kernel(rows, memory, result):
        """Copies each slab's slice out of its window."""
        x = pallas.view(memory, rows[0], tuple(shape), dtype)
        o = pallas.view(result, rows[1], tuple(out), dtype)

        def body(xb, ob):
            """The slice's rows and columns of the window."""
            ob[...] = xb[...][r - r0 : r - r0 + m, c - c0 : c - c0 + n]

        pl.emit_pipeline(body, grid=grid, in_specs=[x_spec], out_specs=[o_spec])(x, o)

    return pallas.program(kernel, spec)
