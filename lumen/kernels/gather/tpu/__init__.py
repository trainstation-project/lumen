from lumen import pallas as pl
from lumen.graph import prims
from lumen.kernels.tpu import pallas

SUBLANES = 8


def _int(n, shape=()):
    return prims.full(shape, n, "int32")


def _picked(values, shape):
    which = pl.broadcasted_iota("int32", shape, 0)
    picked = _int(0, shape)
    for k, v in enumerate(values):
        picked = prims.select(prims.eq(which, _int(k, shape)), prims.broadcast_in_dim(v, shape, ()), picked)
    return picked


def _sparse_core(spec):
    rows, width, count, k, slots = spec["rows_of_view"], spec["width"], spec["count"], spec["k"], spec["slots"]
    cores, subcores = spec["mesh"]
    per = count // (cores * subcores)
    n = per // k
    view, offsets = ("uint32", (rows, width)), ("int32", (count,))

    def kernel(core, subcore, _, src, dst, region, to, *buffers):
        at, block, sems = buffers[:slots], buffers[slots : 2 * slots], buffers[2 * slots :]
        first = (core * subcores + subcore) * per

        def start(i):
            return pl.multiple_of(first + i * k, k)

        def fetch(i, slot):
            pl.async_copy(src.at[pl.ds(start(i), k)], at[slot], sems[slot]).wait()
            pl.async_copy(region.at[at[slot]], block[slot], sems[slot])

        def finish(i, slot):
            pl.make_async_copy(region.at[at[slot]], block[slot], sems[slot]).wait()
            pl.async_copy(dst.at[pl.ds(start(i), k)], to, sems[slot]).wait()
            pl.async_copy(block[slot], region.at[to], sems[slot]).wait()

        if slots == 1:

            @pl.loop(0, n)
            def _(i):
                fetch(i, 0)
                finish(i, 0)

            return
        fetch(0, 0)

        @pl.loop(0, (n - 1) // 2)
        def _(j):
            i = j * 2
            fetch(i + 1, 1)
            finish(i, 0)
            fetch(i + 2, 0)
            finish(i + 1, 1)

        if (n - 1) % 2:
            fetch(n - 1, 1)
            finish(n - 2, 0)
            finish(n - 1, 1)
        else:
            finish(n - 1, 0)

    scratch = [pl.VMEM((k,), "int32")]
    scratch += [pl.VMEM((k,), "int32") for _ in range(slots)]
    scratch += [pl.VMEM((k, width), "uint32") for _ in range(slots)]
    scratch += [pl.SemaphoreType.DMA for _ in range(slots)]
    return pl.sparse_core_kernel(kernel, refs=[view, offsets, offsets, view], mesh=(cores, subcores), scratch=scratch)


def program(spec):
    if spec.get("sparse_core"):
        return _sparse_core(spec)
    vocab, width, dtype, tile = spec["vocab"], spec["width"], spec["dtype"], spec["tile"]
    indices, bm, bn, rw, cw = tuple(spec["indices"]), spec["bm"], spec["bn"], spec["rw"], spec["cw"]
    pack = tile // SUBLANES
    s = indices[-1]
    r = indices[1] if len(indices) == 3 else 1
    lead = (indices[0] if len(indices) == 3 else 1) * r
    grid = (lead, s // bm, width // bn)
    steps = grid[0] * grid[1] * grid[2]
    words = -(-bm // pack)

    def kernel(rows, memory, result):
        table = pallas.view(memory, rows[0], (-(-vocab // pack), width), "int32")
        ids = pallas.view(memory, rows[1], indices, "int32")
        out = pallas.view(result, rows[2], (lead, -(-s // pack), width), "int32")
        sems = pl.scratch(pl.SemaphoreType.DMA((2,)))
        window = pl.scratch(pl.SMEM((2, cw) if len(indices) == 1 else (2, rw, cw), "int32"))
        tiles = pl.scratch(pl.VMEM((2, pack, words * SUBLANES, bn), "int32"))

        def at(step):
            l, i, j = step // (grid[1] * grid[2]), step // grid[2] % grid[1], step % grid[2]
            s0 = pl.multiple_of(i * bm // cw * cw, cw)
            places = [min(k, bm - 1) for k in range(words * pack)]
            if len(indices) == 1:
                return j, ids.at[pl.ds(s0, cw)], [(i * bm + k - s0,) for k in places]
            r0 = pl.multiple_of(l % r // rw * rw, rw)
            return j, ids.at[l // r, pl.ds(r0, rw), pl.ds(s0, cw)], [(l % r - r0, i * bm + k - s0) for k in places]

        def tile_copy(slot, k, start, j):
            columns = pl.ds(pl.multiple_of(j * bn, bn), bn)
            return pl.make_async_copy(
                table.at[pl.ds(start, SUBLANES), columns],
                tiles.at[slot, k % pack, pl.ds(k // pack * SUBLANES, SUBLANES)],
                sems.at[slot],
            )

        def fetch(step, slot):
            j, window_ids, places = at(step)
            c = pl.make_async_copy(window_ids, window.at[slot], sems.at[slot])
            c.start()
            c.wait()
            for k, place in enumerate(places):
                word = window[(slot, *place)] // pack
                tile_copy(slot, k, pl.multiple_of(word // SUBLANES * SUBLANES, SUBLANES), j).start()

        def body(block):
            step = (pl.program_id(0) * grid[1] + pl.program_id(1)) * grid[2] + pl.program_id(2)
            slot = step % 2

            @pl.when(step == 0)
            def _():
                fetch(step, slot)

            for k in range(words * pack):
                tile_copy(slot, k, 0, 0).wait()

            @pl.when(step + 1 < steps)
            def _():
                fetch(step + 1, 1 - slot)

            places = at(step)[2]
            shape, sublane = (words, SUBLANES, 1), pl.broadcasted_iota("int32", (words, SUBLANES, bn), 1)
            value = None
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
