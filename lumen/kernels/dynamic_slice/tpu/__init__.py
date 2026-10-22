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


def _started(copies):
    for c in copies:
        c.start()


def _waited(copies):
    for c in copies:
        c.wait()


def program(spec):
    shape, update, dtype, tile = spec["shape"], spec["update"], spec["dtype"], spec["tile"]
    bm, bn, gm, width = spec["bm"], spec["bn"], spec["gm"], spec["width"]
    copied, cbm, count, fixed = spec["copy"], spec["copy_rows"], spec["count"], spec["static"]
    depth, lag = spec["depth"], 0 if spec["ordered"] else 1
    resident = spec["resident"]
    rank = len(shape)
    (rows, cols), (ur, uc) = shape[-2:], update[-2:]
    rp, cp, urp, ucp = _up(rows, tile), _up(cols, LANES), _up(ur, tile), _up(uc, LANES)
    lead, ulead = shape[:-2], update[:-2]
    whole = (ur % tile == 0 or ur == rows) and (uc % LANES == 0 or uc == cols)
    wide = "float32" if dtype.startswith("bfloat") or dtype.startswith("float") else "int32"
    small = math.prod(ulead) == 1 and gm == urp
    strips = uc != cols and width > 2 * STRIP
    blocks = urp // gm
    indices = 0 if fixed else count * rank
    output = 1 + count + indices

    def kernel(refs, memory, result):
        sem = pl.scratch(pl.SemaphoreType.DMA)
        x = pallas.view(memory, refs[0], (math.prod(lead), rp, cp), dtype)
        out = pallas.view(result, refs[output], (math.prod(lead), rp, cp), dtype)
        slots = pl.scratch(pl.VMEM((SLOTS, bm, bn), dtype))
        flights = pl.scratch(pl.SemaphoreType.DMA((2, SLOTS)))
        sources = pl.scratch(pl.VMEM((depth, gm, ucp), dtype))
        windows = pl.scratch(pl.VMEM((depth, gm, width), dtype))
        edge = pl.scratch(pl.VMEM((tile, width), dtype))
        loads, reads, stores = (pl.scratch(pl.SemaphoreType.DMA((depth,))) for _ in range(3))

        def updated(m):
            return pallas.view(memory, refs[1 + m], (math.prod(ulead), urp, ucp), dtype)

        def source(m, k, i, slot):
            rows_at = pl.multiple_of(i * gm, tile)
            return pl.make_async_copy(updated(m).at[k, pl.ds(rows_at, gm)], sources.at[slot], loads.at[slot])

        if small:
            for j in range(min(depth, count)):
                source(_int(j), 0, _int(0), _int(j)).start()

        if copied:

            def moved(i, o):
                o[...] = i[...]

            whole_rows = pl.BlockSpec((None, cbm, cp), lambda l, i: (l, i, 0))
            pl.emit_pipeline(moved, grid=(math.prod(lead), rp // cbm), in_specs=[whole_rows], out_specs=[whole_rows])(
                x, out
            )

        if fixed:

            def start(m, d):
                return refs[output + 1 + m * rank + d]

        else:
            held = pl.scratch(pl.SMEM((indices, SCALARS), "int32"))
            fetched = [
                pl.async_copy(pallas.view(memory, refs[1 + count + j], (SCALARS,), "int32"), held.at[j], sem)
                for j in range(indices)
            ]
            _waited(fetched)

            def start(m, d):
                return prims.min(prims.max(held[m * rank + d, 0], _int(0)), _int(shape[d] - update[d]))

        def one(m, waiting=False):
            starts = [start(m, d) for d in range(rank)]
            r0, c0 = starts[-2], starts[-1]
            a = pl.multiple_of(r0 // tile * tile, tile)
            b = pl.multiple_of(prims.min(c0 // LANES * LANES, _int(cp - width)), LANES)
            dr, dc = r0 - a, c0 - b
            aligned = (dr == 0) & (c0 % LANES == 0)

            def slab(k):
                at, rest = _int(0), k
                for d in reversed(range(len(ulead))):
                    at = at + (starts[d] + rest % ulead[d]) * math.prod(lead[d + 1 :])
                    rest = rest // ulead[d]
                return at

            def piece(k, t):
                across = ucp // bn
                rows_at, cols_at = pl.multiple_of(t // across * bm, tile), pl.multiple_of(t % across * bn, LANES)
                return pl.make_async_copy(
                    updated(m).at[k, pl.ds(rows_at, bm), pl.ds(cols_at, bn)],
                    slots.at[t % SLOTS],
                    flights.at[0, t % SLOTS],
                )

            def staged(k, at):
                total = urp // bm * (ucp // bn)
                for t in range(min(SLOTS, total)):
                    piece(k, _int(t)).start()

                def store(t):
                    across = ucp // bn
                    rows_at = pl.multiple_of(r0 + t // across * bm, tile)
                    cols_at = pl.multiple_of(c0 + t % across * bn, LANES)
                    to = out.at[at, pl.ds(rows_at, bm), pl.ds(cols_at, bn)]
                    return pl.make_async_copy(slots.at[t % SLOTS], to, flights.at[1, t % SLOTS])

                @pl.loop(0, total)
                def _(t):
                    piece(k, t).wait()
                    store(t).start()

                    @pl.when(t + SLOTS < total)
                    def _():
                        store(t).wait()
                        piece(k, t + SLOTS).start()

                @pl.loop(max(0, total - SLOTS), total)
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

            def window_reads(at, slot, top):
                to = windows.at[slot]
                target = out.at[at, pl.ds(top, gm)]
                full = pl.make_async_copy(target.at[:, pl.ds(b, width)], to, reads.at[slot])
                if not strips:
                    return [full], [full] if uc != cols else []
                left = pl.make_async_copy(target.at[:, pl.ds(b, STRIP)], to.at[:, pl.ds(0, STRIP)], reads.at[slot])
                right_at = pl.multiple_of(b + width - STRIP, LANES)
                right = pl.make_async_copy(
                    target.at[:, pl.ds(right_at, STRIP)], to.at[:, pl.ds(width - STRIP, STRIP)], reads.at[slot]
                )
                return [full], [left, right]

            def edged(at, last):
                @pl.when(dr + ur > urp)
                def _():
                    top = pl.multiple_of(a + urp, tile)
                    target = out.at[at, pl.ds(top, tile), pl.ds(b, width)]
                    pl.async_copy(target, edge, sem).wait()
                    tail = sources[last, gm - tile : gm, :]
                    v = placed(tail, prims.full((tile, ucp), 0, dtype), tile)
                    edge[...] = prims.select(inside(top, tile), v, edge[...].to(wide)).to(dtype)
                    pl.async_copy(edge, target, sem).wait()

            def block(at, i, slot, tail_slot, loaded, prefetch):
                top = pl.multiple_of(a + i * gm, tile)
                bordered = (top < r0) | (top + gm > r0 + ur)
                whole_copies, strip_copies = window_reads(at, slot, top)
                pl.when(bordered)(lambda: _started(whole_copies))
                pl.when(~bordered)(lambda: _started(strip_copies))
                loaded()
                tail = sources[tail_slot, gm - tile : gm, :]
                current = sources[slot, :, :]
                prefetch()
                v = placed(tail, current, gm)
                pl.when(bordered)(lambda: _waited(whole_copies))
                pl.when(~bordered)(lambda: _waited(strip_copies))
                to = windows.at[slot]
                to[...] = prims.select(inside(top, gm), v, to[...].to(wide)).to(dtype)
                return pl.make_async_copy(to, out.at[at, pl.ds(top, gm), pl.ds(b, width)], stores.at[slot])

            def streamed(k, at):
                source(m, k, _int(0), _int(0)).start()

                def written(i):
                    top = pl.multiple_of(a + i * gm, tile)
                    return pl.make_async_copy(
                        windows.at[i % 2], out.at[at, pl.ds(top, gm), pl.ds(b, width)], stores.at[i % 2]
                    )

                @pl.loop(0, blocks)
                def _(i):
                    @pl.when(i >= 2)
                    def _():
                        written(i - 2).wait()

                    def prefetch():
                        @pl.when(i + 1 < blocks)
                        def _():
                            source(m, k, i + 1, (i + 1) % 2).start()

                    block(at, i, i % 2, (i + 1) % 2, lambda: source(m, k, i, i % 2).wait(), prefetch).start()

                @pl.loop(max(0, blocks - 2), blocks)
                def _(i):
                    written(i).wait()

                edged(at, (blocks - 1) % 2)

            if small:
                slot = m % depth
                at = slab(0)
                to = out.at[at, pl.ds(pl.multiple_of(r0, tile), gm), pl.ds(pl.multiple_of(c0, LANES), ucp)]
                fast_store = pl.make_async_copy(sources.at[slot], to, stores.at[slot])
                top = pl.multiple_of(a, tile)
                slow_store = pl.make_async_copy(
                    windows.at[slot], out.at[at, pl.ds(top, gm), pl.ds(b, width)], stores.at[slot]
                )
                if waiting:
                    if whole:
                        pl.when(aligned)(fast_store.wait)
                        pl.when(~aligned)(slow_store.wait)
                    else:
                        slow_store.wait()
                    return

                def slow():
                    block(at, _int(0), slot, (m + 1) % depth, lambda: None, lambda: None).start()
                    edged(at, slot)

                source(m, 0, _int(0), slot).wait()
                if whole:
                    pl.when(aligned)(fast_store.start)
                    pl.when(~aligned)(slow)
                else:
                    slow()
                return

            @pl.loop(0, math.prod(ulead))
            def _(k):
                at = slab(k)
                if not whole:
                    streamed(k, at)
                    return
                pl.when(aligned)(lambda: staged(k, at))
                pl.when(~aligned)(lambda: streamed(k, at))

        def refill(j):
            @pl.when(j + depth < count)
            def _():
                source(j + depth, 0, _int(0), j % depth).start()

        if resident:
            top, bottom, left, right, at = resident
            height, across = bottom - top, right - left
            tall, wide_as = min(urp + tile, height), min(width, across)
            kept_box = pl.scratch(pl.VMEM((height, across), dtype))
            box = out.at[at, pl.ds(top, height), pl.ds(left, across)]
            pl.async_copy(box, kept_box, sem).wait()

            def kept(m):
                r0, c0 = start(m, rank - 2) - top, start(m, rank - 1) - left
                ra = pl.multiple_of(prims.min(r0 // tile * tile, _int(height - tall)), tile)
                cb = pl.multiple_of(prims.min(c0 // LANES * LANES, _int(across - wide_as)), LANES)
                dr, dc = r0 - ra, c0 - cb
                slot = m % depth
                source(m, 0, _int(0), slot).wait()

                def fast():
                    rows_at, cols_at = pl.multiple_of(r0, tile), pl.multiple_of(c0, LANES)
                    kept_box[pl.ds(rows_at, gm), pl.ds(cols_at, ucp)] = sources[slot, :, :]

                def slow():
                    v = sources[slot, :, :].to(wide)
                    if tall > gm:
                        v = prims.concatenate([v, prims.full((tall - gm, ucp), 0, wide)], 0)
                    if wide_as > ucp:
                        v = prims.concatenate([v, prims.full((tall, wide_as - ucp), 0, wide)], 1)
                    v = pl.roll(pl.roll(v, dr, 0), dc, 1)
                    g = pl.broadcasted_iota("int32", (tall, wide_as), 0)
                    c = pl.broadcasted_iota("int32", (tall, wide_as), 1)
                    inside = (g >= dr) & (g < dr + ur) & (c >= dc) & (c < dc + uc)
                    window = kept_box.at[pl.ds(ra, tall), pl.ds(cb, wide_as)]
                    window[...] = prims.select(inside, v, window[...].to(wide)).to(dtype)

                if whole:
                    aligned = (r0 % tile == 0) & (c0 % LANES == 0)
                    pl.when(aligned)(fast)
                    pl.when(~aligned)(slow)
                else:
                    slow()
                refill(m)

            @pl.loop(0, count)
            def _(m):
                kept(m)

            pl.async_copy(kept_box, box, sem).wait()
            return

        @pl.loop(0, count)
        def _(m):
            one(m)
            if not small:
                return
            if lag == 0:
                one(m, waiting=True)
                refill(m)
                return

            @pl.when(m >= 1)
            def _():
                one(m - 1, waiting=True)
                refill(m - 1)

        if small and lag:
            one(_int(count - 1), waiting=True)

    return pallas.program(kernel, spec)
