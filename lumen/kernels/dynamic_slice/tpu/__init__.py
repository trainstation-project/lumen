import math

from lumen import pallas as pl
from lumen.graph import prims
from lumen.kernels.tpu import pallas

LANES = 128

SCALARS = 128

SLOTS = 2
STRIP = 2 * LANES


def _up(n, m):
    return -(-n // m) * m


def _int(n):
    return prims.full((), n, "int32")


def program(spec):
    shape, update, dtype, tile = spec["shape"], spec["update"], spec["dtype"], spec["tile"]
    bm, bn, gm, width = spec["bm"], spec["bn"], spec["gm"], spec["width"]
    copied, cbm = spec["copy"], spec["copy_rows"]
    rank = len(shape)
    (rows, cols), (ur, uc) = shape[-2:], update[-2:]
    rp, cp, urp, ucp = _up(rows, tile), _up(cols, LANES), _up(ur, tile), _up(uc, LANES)
    lead, ulead = shape[:-2], update[:-2]
    whole = (ur % tile == 0 or ur == rows) and (uc % LANES == 0 or uc == cols)
    wide = "float32" if dtype.startswith("bfloat") or dtype.startswith("float") else "int32"
    single = math.prod(ulead) == 1 and urp == bm and ucp == bn
    strips = uc != cols and width > 2 * STRIP
    blocks = urp // gm

    def kernel(refs, memory, result):
        sem = pl.scratch(pl.SemaphoreType.DMA)
        x = pallas.view(memory, refs[0], (math.prod(lead), rp, cp), dtype)
        u = pallas.view(memory, refs[1], (math.prod(ulead), urp, ucp), dtype)
        out = pallas.view(result, refs[2 + rank], (math.prod(lead), rp, cp), dtype)
        slots = pl.scratch(pl.VMEM((SLOTS, bm, bn), dtype))
        flights = pl.scratch(pl.SemaphoreType.DMA((2, SLOTS)))
        sources = pl.scratch(pl.VMEM((2, gm, ucp), dtype))
        windows = pl.scratch(pl.VMEM((2, gm, width), dtype))
        edge = pl.scratch(pl.VMEM((tile, width), dtype))
        loads, reads, stores = (pl.scratch(pl.SemaphoreType.DMA((2,))) for _ in range(3))

        def piece(k, t):
            across = ucp // bn
            a, b = pl.multiple_of(t // across * bm, tile), pl.multiple_of(t % across * bn, LANES)
            return pl.make_async_copy(u.at[k, pl.ds(a, bm), pl.ds(b, bn)], slots.at[t % SLOTS], flights.at[0, t % SLOTS])

        def source(k, i):
            rows_at = pl.multiple_of(i * gm, tile)
            return pl.make_async_copy(u.at[k, pl.ds(rows_at, gm)], sources.at[i % 2], loads.at[i % 2])

        early = source(0, _int(0)) if math.prod(ulead) == 1 else None
        if early is not None:
            early.start()

        if copied:

            def moved(i, o):
                o[...] = i[...]

            whole_rows = pl.BlockSpec((None, cbm, cp), lambda l, i: (l, i, 0))
            pl.emit_pipeline(moved, grid=(math.prod(lead), rp // cbm), in_specs=[whole_rows], out_specs=[whole_rows])(
                x, out
            )
        held = pl.scratch(pl.SMEM((rank, SCALARS), "int32"))
        fetched = [
            pl.async_copy(pallas.view(memory, refs[2 + d], (SCALARS,), "int32"), held.at[d], sem) for d in range(rank)
        ]
        for c in fetched:
            c.wait()
        starts = [prims.min(prims.max(held[d, 0], _int(0)), _int(shape[d] - update[d])) for d in range(rank)]
        r0, c0 = starts[-2], starts[-1]
        a = pl.multiple_of(r0 // tile * tile, tile)
        b = pl.multiple_of(prims.min(c0 // LANES * LANES, _int(cp - width)), LANES)
        dr, dc = r0 - a, c0 - b

        def slab(k):
            at, rest = _int(0), k
            for d in reversed(range(len(ulead))):
                at = at + (starts[d] + rest % ulead[d]) * math.prod(lead[d + 1 :])
                rest = rest // ulead[d]
            return at

        def staged(k, at):
            if early is not None:
                early.wait()
            if single:
                to = out.at[at, pl.ds(pl.multiple_of(r0, tile), bm), pl.ds(pl.multiple_of(c0, LANES), bn)]
                pl.async_copy(sources.at[0], to, sem).wait()
                return
            count = urp // bm * (ucp // bn)
            for t in range(min(SLOTS, count)):
                piece(k, _int(t)).start()

            def store(t):
                across = ucp // bn
                rows_at = pl.multiple_of(r0 + t // across * bm, tile)
                cols_at = pl.multiple_of(c0 + t % across * bn, LANES)
                to = out.at[at, pl.ds(rows_at, bm), pl.ds(cols_at, bn)]
                return pl.make_async_copy(slots.at[t % SLOTS], to, flights.at[1, t % SLOTS])

            @pl.loop(0, count)
            def _(t):
                piece(k, t).wait()
                store(t).start()

                @pl.when(t + SLOTS < count)
                def _():
                    store(t).wait()
                    piece(k, t + SLOTS).start()

            @pl.loop(max(0, count - SLOTS), count)
            def _(t):
                store(t).wait()

        def placed(tail, current, rows_of):
            v = prims.concatenate([tail, current], 0).to(wide)
            if width > ucp:
                v = prims.concatenate([v, prims.full((tile + rows_of, width - ucp), 0, wide)], 1)
            v = pl.roll(pl.roll(v, dr, 0), dc, 1)
            return prims.slice(v, (tile, 0), (tile + rows_of, width))

        def inside(top, rows_of):
            g = top + pl.broadcasted_iota("int32", (rows_of, width), 0)
            c = pl.broadcasted_iota("int32", (rows_of, width), 1)
            return (g >= r0) & (g < r0 + ur) & (c >= dc) & (c < dc + uc)

        def window_reads(at, i, top):
            to = windows.at[i % 2]
            target = out.at[at, pl.ds(top, gm)]
            full = pl.make_async_copy(target.at[:, pl.ds(b, width)], to, reads.at[i % 2])
            if not strips:
                return [full], [full] if uc != cols else None
            left = pl.make_async_copy(target.at[:, pl.ds(b, STRIP)], to.at[:, pl.ds(0, STRIP)], reads.at[i % 2])
            right_at = pl.multiple_of(b + width - STRIP, LANES)
            right = pl.make_async_copy(
                target.at[:, pl.ds(right_at, STRIP)], to.at[:, pl.ds(width - STRIP, STRIP)], reads.at[i % 2]
            )
            return [full], [left, right]

        def streamed(k, at):
            if early is None:
                source(k, _int(0)).start()

            def written(i):
                top = pl.multiple_of(a + i * gm, tile)
                return pl.make_async_copy(windows.at[i % 2], out.at[at, pl.ds(top, gm), pl.ds(b, width)], stores.at[i % 2])

            @pl.loop(0, blocks)
            def _(i):
                top = pl.multiple_of(a + i * gm, tile)
                bordered = (top < r0) | (top + gm > r0 + ur)

                @pl.when(i >= 2)
                def _():
                    written(i - 2).wait()

                whole_copies, strip_copies = window_reads(at, i, top)

                @pl.when(bordered)
                def _():
                    for c in whole_copies:
                        c.start()

                if strip_copies:

                    @pl.when(~bordered)
                    def _():
                        for c in strip_copies:
                            c.start()

                source(k, i).wait()
                tail = sources[(i + 1) % 2, gm - tile : gm, :]
                current = sources[i % 2, :, :]

                @pl.when(i + 1 < blocks)
                def _():
                    source(k, i + 1).start()

                v = placed(tail, current, gm)

                @pl.when(bordered)
                def _():
                    for c in whole_copies:
                        c.wait()

                if strip_copies:

                    @pl.when(~bordered)
                    def _():
                        for c in strip_copies:
                            c.wait()

                to = windows.at[i % 2]
                to[...] = prims.select(inside(top, gm), v, to[...].to(wide)).to(dtype)
                written(i).start()

            @pl.loop(max(0, blocks - 2), blocks)
            def _(i):
                written(i).wait()

            @pl.when(dr + ur > urp)
            def _():
                top = pl.multiple_of(a + urp, tile)
                target = out.at[at, pl.ds(top, tile), pl.ds(b, width)]
                pl.async_copy(target, edge, sem).wait()
                tail = sources[(blocks - 1) % 2, gm - tile : gm, :]
                v = placed(tail, prims.full((tile, ucp), 0, dtype), tile)
                edge[...] = prims.select(inside(top, tile), v, edge[...].to(wide)).to(dtype)
                pl.async_copy(edge, target, sem).wait()

        @pl.loop(0, math.prod(ulead))
        def _(k):
            at = slab(k)
            if not whole:
                streamed(k, at)
                return
            aligned = (dr == 0) & (c0 % LANES == 0)
            pl.when(aligned)(lambda: staged(k, at))
            pl.when(~aligned)(lambda: streamed(k, at))

    return pallas.program(kernel, spec)
