"""Pallas `take_along_axis` for TPU: each output element is the operand's at the same coordinates but the index along
the axis, read where the operand lies in its TPU tiles (it is never flattened)."""

import lumen
import lumen.functional as F
from lumen.codegen.devices.tpu import pallas
from lumen.kernels import pallas as pl

LANES = 128


def _int(n, shape):
    """Int32 `n` (a Python int or a traced scalar) broadcast to `shape`."""
    n = lumen.full((), n, "int32") if isinstance(n, int) else n
    return n.expand(tuple(shape))


def _slab(l, lead, strides, axis=None, at=None):
    """The operand's slab (flat leading index) for the indices' slab `l`: the same leading coordinates (`lead`, the
    indices' leading dims; `strides`, the operand's row-major slab strides), the one at `axis` replaced by `at`."""
    slab, step = 0, 1
    for d in reversed(range(len(lead))):
        coordinate = at if d == axis else l // step % lead[d]
        slab = slab + coordinate * strides[d]
        step *= lead[d]
    return slab


def _picked(x, local, size, axis):
    """`x` (one register of `size` along `axis`) gathered at `local`, an index into it clamped into range, in the
    operand's index width (Mosaic gathers with indices as wide as the elements)."""
    clamped = F.minimum(F.maximum(local, _int(0, local.shape)), _int(size - 1, local.shape))
    return F.gather(x, axis, clamped)


def program(spec):
    """The `take_along_axis` kernel of `spec`: the operand viewed `[slabs, m, n]` (`x`) and the int32 indices and
    output `[slabs, mi, ni]` (`idx`), both at their tile-padded shapes; the indices' leading dims `lead`, the
    operand's slab strides `strides`; `bm` x `bn` output blocks. By `kind`:

    - ``lanes`` (the last axis): each block reads its rows of the operand whole and gathers them a register of
      `LANES` columns at a time, keeping each element from the register its index falls in.
    - ``sublanes`` (the second-last axis): each block reads its columns of the operand whole and gathers them a
      register of `rows` rows at a time the same way.
    - ``outer`` (an earlier axis, `axis` of the leading dims, `n` long): each block reads the operand's block at
      every position `k` along the axis and keeps it where the index is `k`.

    Its scalars are the operand's row, the indices', then the output's."""
    kind, dtype = spec["kind"], spec["dtype"]
    (slabs, m, n), (islabs, mi, ni) = spec["x"], spec["idx"]
    lead, strides, bm, bn = spec["lead"], spec["strides"], spec["bm"], spec["bn"]
    grid = (islabs, mi // bm, ni // bn)
    block = pl.BlockSpec((None, bm, bn), lambda l, i, j: (l, i, j))

    def kernel(rows, memory, result):
        """Fills each output block from the operand's tiles its indices pick."""
        x = pallas.view(memory, rows[0], (slabs, m, n), dtype)
        idx = pallas.view(memory, rows[1], (islabs, mi, ni), "int32")
        out = pallas.view(result, rows[2], (islabs, mi, ni), dtype)

        if kind == "lanes":
            whole = pl.BlockSpec((None, bm, n), lambda l, i, j: (_slab(l, lead, strides), i, 0))

            def body(ib, xb, ob):
                """Gathers the block's rows a register of columns at a time."""
                at = ib[...]

                def chunk(c, acc):
                    """Takes the elements whose index falls in register `c`."""
                    base = pl.multiple_of(c * LANES, LANES)
                    local = at - _int(base, at.shape)
                    hit = (local >= _int(0, at.shape)) & (local < _int(LANES, at.shape))
                    return F.where(hit, _picked(xb[:, pl.ds(base, LANES)], local, LANES, 1), acc)

                ob[...] = pl.fori_loop(0, n // LANES, chunk, lumen.zeros((bm, bn), dtype))

            pl.emit_pipeline(body, grid=grid, in_specs=[block, whole], out_specs=[block])(idx, x, out)
        elif kind == "sublanes":
            tile = spec["tile"]
            whole = pl.BlockSpec((None, m, bn), lambda l, i, j: (_slab(l, lead, strides), 0, j))

            def body(ib, xb, ob):
                """Gathers the block's columns a register of rows at a time."""
                at = ib[...]

                def chunk(c, acc):
                    """Takes the elements whose index falls in register `c`."""
                    base = pl.multiple_of(c * tile, tile)
                    local = at - _int(base, at.shape)
                    hit = (local >= _int(0, at.shape)) & (local < _int(tile, at.shape))
                    return F.where(hit, _picked(xb[pl.ds(base, tile), pl.ds(0, bn)], local, tile, 0), acc)

                ob[...] = pl.fori_loop(0, m // tile, chunk, lumen.zeros((bm, bn), dtype))

            pl.emit_pipeline(body, grid=grid, in_specs=[block, whole], out_specs=[block])(idx, x, out)
        else:
            axis, length = spec["axis"], spec["n"]
            sem = pl.scratch(pl.SemaphoreType.DMA(()))
            held = pl.scratch(pl.VMEM((bm, bn), dtype))

            def body(ib, ob):
                """Keeps the operand's block at each position along the axis where the index is that position."""
                l, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
                at = ib[...]

                def position(k, acc):
                    """Fetches the operand's block at position `k` and keeps it where the index is `k`."""
                    slab = _slab(l, lead, strides, axis, k)
                    rows_, columns = pl.ds(pl.multiple_of(i * bm, bm), bm), pl.ds(pl.multiple_of(j * bn, bn), bn)
                    c = pl.make_async_copy(x.at[slab, rows_, columns], held, sem)
                    c.start()
                    c.wait()
                    return F.where(at == _int(k, at.shape), held[...], acc)

                ob[...] = pl.fori_loop(0, length, position, lumen.zeros((bm, bn), dtype))

            pl.emit_pipeline(body, grid=grid, in_specs=[block], out_specs=[block])(idx, out)

    return pallas.program(kernel, spec)
