"""Pallas reshapes for TPU that split or merge the last dimension of a TPU-tiled tensor."""

from lumen import pallas as pl
from lumen.codegen.utils.tpu import pallas


def program(spec):
    """A reshape that splits (``split``) a last dimension of ``a`` times
    ``w`` into ``[a, w]``, or merges ``[a, w]`` into one: ``lead`` slabs of
    ``m`` rows, ``bm`` rows a block, the ``[a, w]`` side ``lead * m`` slabs
    of ``a`` rows. Its scalars are the operand's row, then the result's."""
    lead, m, a, w, bm, dtype = (spec[k] for k in ("lead", "m", "a", "w", "bm", "dtype"))
    blocks = m // bm
    flat = pl.BlockSpec((None, bm, a * w), lambda l, i: (l, i, 0))
    tiled = pl.BlockSpec((bm, a, w), lambda l, i: (l * blocks + i, 0, 0))

    def kernel(rows, memory, result):
        """Copies blocks between the flat `[.., m, a * w]` and tiled `[.., a, w]` forms."""
        flat_shape, tiled_shape = (lead, m, a * w), (lead * m, a, w)
        if spec["split"]:
            x = pallas.view(memory, rows[0], flat_shape, dtype)
            y = pallas.view(result, rows[1], tiled_shape, dtype)

            def body(xb, yb):
                """Splits each block's columns into `a` slices of `w`."""
                v = xb[...]
                for k in range(a):
                    yb[:, k, :] = v[:, k * w : (k + 1) * w]

            pl.emit_pipeline(body, grid=(lead, blocks), in_specs=[flat], out_specs=[tiled])(x, y)
            return
        x = pallas.view(memory, rows[0], tiled_shape, dtype)
        y = pallas.view(result, rows[1], flat_shape, dtype)

        def body(xb, yb):
            """Merges each block's `a` slices of `w` back into columns."""
            for k in range(a):
                yb[:, k * w : (k + 1) * w] = xb[:, k, :]

        pl.emit_pipeline(body, grid=(lead, blocks), in_specs=[tiled], out_specs=[flat])(x, y)

    return pallas.program(kernel, spec)
