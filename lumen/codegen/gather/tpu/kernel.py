"""Pallas row gather for TPU: a TensorCore kernel over TPU-tiled tables and a SparseCore DMA kernel."""

from lumen import pallas as pl
from lumen.graph import prims
from lumen.kernels.tpu import pallas

SUBLANES = 8


def _int(n, shape=()):
    """Traced int32 constant `n` of `shape`."""
    return prims.full(shape, n, "int32")


def _picked(values, shape):
    """Tensor of `shape` holding `values[k]` at row `k` of its leading dim."""
    which = pl.broadcasted_iota("int32", shape, 0)
    picked = _int(0, shape)
    for k, v in enumerate(values):
        picked = prims.select(prims.eq(which, _int(k, shape)), prims.broadcast_in_dim(v, shape, ()), picked)
    return picked


def _sparse_core(spec):
    """SparseCore gather: each subcore gathers its share of indices `k` rows at a time by indirect DMA,
    then scatters them to explicit destination rows or, when `linear`, to consecutive output rows."""
    rows, width, count, k, n = spec["rows_of_view"], spec["width"], spec["count"], spec["k"], spec["n"]
    linear, scalars = spec["linear"], spec["scalars"]
    cores, subcores = spec["mesh"]
    per = count // (cores * subcores)
    view, offsets = ("uint32", (rows, width)), ("int32", (count,))
    last = n // k * k
    tail = -(-(n - last) // 8) * 8

    def kernel(core, subcore, _, src, dst_or_rows, region, at, block, sem, *buffers):
        """Gathers this worker's `per` indices from `region` into the output."""
        first = (core * subcores + subcore) * per
        if linear:
            shared, scalar = buffers
            pl.async_copy(dst_or_rows, shared, sem).wait()
            pl.async_copy(shared, scalar, sem).wait()
            # The output's first row, in units of `width`-word rows, read from the scalar operands.
            base = scalar[spec["out"]] // (width // 128)

        @pl.loop(0, per // k)
        def _(i):
            """Moves chunk `i` of `k` indices: their rows into `block`, then out to the destination."""
            start = pl.multiple_of(first + i * k, k)
            pl.async_copy(src.at[pl.ds(start, k)], at, sem).wait()
            pl.async_copy(region.at[at], block, sem).wait()
            if not linear:
                pl.async_copy(dst_or_rows.at[pl.ds(start, k)], at, sem).wait()
                pl.async_copy(block, region.at[at], sem).wait()
                return
            row = pl.multiple_of(base + start, 8)

            @pl.when(start + k <= n)
            def _():
                """Writes a full chunk to its consecutive output rows."""
                pl.async_copy(block, region.at[pl.ds(row, k)], sem).wait()

            # The last chunk writes only the rows up to `n`, rounded up to whole sublane tiles.
            if tail:

                @pl.when(start == last)
                def _():
                    """Writes the partial last chunk."""
                    pl.async_copy(block.at[pl.ds(0, tail)], region.at[pl.ds(row, tail)], sem).wait()

    second = ("int32", (scalars,)) if linear else offsets
    scratch = [pl.VMEM((k,), "int32"), pl.VMEM((k, width), "uint32"), pl.SemaphoreType.DMA]
    if linear:
        scratch += [pl.VMEM_SHARED((scalars,), "int32"), pl.SMEM((scalars,), "int32")]
    return pl.sparse_core_kernel(kernel, refs=[view, offsets, second, view], mesh=(cores, subcores), scratch=scratch)


def program(spec):
    """Builds the gather kernel for `spec`: SparseCore when requested, otherwise a TensorCore kernel
    that DMAs the tile holding each wanted row and picks the row's sublane (and half-word) out of it."""
    if spec.get("sparse_core"):
        return _sparse_core(spec)
    vocab, width, dtype, tile = spec["vocab"], spec["width"], spec["dtype"], spec["tile"]
    indices, bm, bn, rw, cw = tuple(spec["indices"]), spec["bm"], spec["bn"], spec["rw"], spec["cw"]
    # `pack` table rows share each 32-bit word; a fetched tile spans `SUBLANES` words of rows.
    pack = tile // SUBLANES
    s = indices[-1]
    r = indices[1] if len(indices) == 3 else 1
    lead = (indices[0] if len(indices) == 3 else 1) * r
    grid = (lead, s // bm, width // bn)
    steps = grid[0] * grid[1] * grid[2]
    words = -(-bm // pack)

    def kernel(rows, memory, result):
        """Gathers blocks of `bm` rows by `bn` columns, double-buffering the next block's tile fetches."""
        table = pallas.view(memory, rows[0], (-(-vocab // pack), width), "int32")
        ids = pallas.view(memory, rows[1], indices, "int32")
        out = pallas.view(result, rows[2], (lead, -(-s // pack), width), "int32")
        sems = pl.scratch(pl.SemaphoreType.DMA((2,)))
        window = pl.scratch(pl.SMEM((2, cw) if len(indices) == 1 else (2, rw, cw), "int32"))
        tiles = pl.scratch(pl.VMEM((2, pack, words * SUBLANES, bn), "int32"))

        def at(step):
            """Column block, index window and per-row positions in the window for flat pipeline `step`."""
            l, i, j = step // (grid[1] * grid[2]), step // grid[2] % grid[1], step % grid[2]
            s0 = pl.multiple_of(i * bm // cw * cw, cw)
            places = [min(k, bm - 1) for k in range(words * pack)]
            if len(indices) == 1:
                return j, ids.at[pl.ds(s0, cw)], [(i * bm + k - s0,) for k in places]
            r0 = pl.multiple_of(l % r // rw * rw, rw)
            return j, ids.at[l // r, pl.ds(r0, rw), pl.ds(s0, cw)], [(l % r - r0, i * bm + k - s0) for k in places]

        def tile_copy(slot, k, start, j):
            """DMA of the 8-sublane tile starting at word row `start` into slot `slot` for output row `k`."""
            columns = pl.ds(pl.multiple_of(j * bn, bn), bn)
            return pl.make_async_copy(
                table.at[pl.ds(start, SUBLANES), columns],
                tiles.at[slot, k % pack, pl.ds(k // pack * SUBLANES, SUBLANES)],
                sems.at[slot],
            )

        def fetch(step, slot):
            """Loads `step`'s index window and starts fetching the tile of every row it needs into `slot`."""
            j, window_ids, places = at(step)
            c = pl.make_async_copy(window_ids, window.at[slot], sems.at[slot])
            c.start()
            c.wait()
            for k, place in enumerate(places):
                word = window[(slot, *place)] // pack
                tile_copy(slot, k, pl.multiple_of(word // SUBLANES * SUBLANES, SUBLANES), j).start()

        def body(block):
            """Assembles one output block from the fetched tiles, prefetching the next step's tiles."""
            step = (pl.program_id(0) * grid[1] + pl.program_id(1)) * grid[2] + pl.program_id(2)
            slot = step % 2

            @pl.when(step == 0)
            def _():
                """Fetches the first step's tiles."""
                fetch(step, slot)

            for k in range(words * pack):
                tile_copy(slot, k, 0, 0).wait()

            @pl.when(step + 1 < steps)
            def _():
                """Prefetches the next step's tiles into the other slot."""
                fetch(step + 1, 1 - slot)

            places = at(step)[2]
            shape, sublane = (words, SUBLANES, 1), pl.broadcasted_iota("int32", (words, SUBLANES, bn), 1)
            value = None
            # Pick each row's sublane out of its tile; packed dtypes then shift out the row's 16-bit half.
            for p in range(pack):
                rows_of = [window[(slot, *place)] for place in places[p::pack]]
                within = _picked([row // pack % SUBLANES for row in rows_of], shape)
                mine = prims.eq(sublane, prims.broadcast_in_dim(within, (words, SUBLANES, bn), (0, 1, 2)))
                bits = prims.reshape(tiles[slot, p], (words, SUBLANES, bn))
                part = prims.reduce_sum(prims.select(mine, bits, _int(0, (words, SUBLANES, bn))), (1,), "int32")
                if pack > 1:
                    half = _picked([row % pack * 16 for row in rows_of], (words, 1))
                    half = prims.broadcast_in_dim(half, (words, bn), (0, 1))
                    part = pl.bitwise_and(pl.shift_right_logical(part, half), _int(0xFFFF, (words, bn)))
                    part = pl.shift_left(part, _int(16 * p, (words, bn)))
                value = part if value is None else pl.bitwise_or(value, part)
            block[...] = value

        spec_out = pl.BlockSpec((None, words, bn), lambda l, i, j: (l, i, j))
        pl.emit_pipeline(body, grid=grid, in_specs=[], out_specs=[spec_out])(out)

    return pallas.program(kernel, spec)
