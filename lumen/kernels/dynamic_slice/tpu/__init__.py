"""Pallas TPU kernel for `dynamic_update_slice`, writing one or more updates into a copy of the operand.

Unaligned updates are rolled into place inside tile-aligned windows; aligned ones are DMA'd directly.
"""

import math

from lumen import pallas as pl
from lumen.graph import prims
from lumen.kernels.tpu import pallas

# Lane width of a TPU tile.
LANES = 128

# Length of the SMEM row holding each fetched start index.
SCALARS = 128

# Buffers for pipelining the staged pieces of an aligned update.
SLOTS = 2
# Column strip width read at each side of an interior window.
STRIP = 2 * LANES


def _up(n, m):
    """Rounds `n` up to a multiple of `m`."""
    return -(-n // m) * m


def _int(n):
    """Scalar int32 constant `n`."""
    return prims.full((), n, "int32")


def _started(copies):
    """Starts every DMA in `copies`."""
    for c in copies:
        c.start()


def _waited(copies):
    """Waits on every DMA in `copies`."""
    for c in copies:
        c.wait()


def program(spec):
    """Builds the Pallas program for the dynamic_update_slice plan in `spec` (see the Rust `encode`)."""
    shape, update, dtype, tile = spec["shape"], spec["update"], spec["dtype"], spec["tile"]
    bm, bn, gm, width = spec["bm"], spec["bn"], spec["gm"], spec["width"]
    copied, count, fixed = spec["copy"], spec["count"], spec["static"]
    # Overlapping updates (`ordered`) must finish before the next starts; otherwise one may lag.
    depth, lag = spec["depth"], 0 if spec["ordered"] else 1
    resident = spec["resident"]
    rank = len(shape)
    (rows, cols), (ur, uc) = shape[-2:], update[-2:]
    rp, cp, urp, ucp = _up(rows, tile), _up(cols, LANES), _up(ur, tile), _up(uc, LANES)
    lead, ulead = shape[:-2], update[:-2]
    whole = (ur % tile == 0 or ur == rows) and (uc % LANES == 0 or uc == cols)
    wide = "float32" if dtype.startswith("bfloat") or dtype.startswith("float") else "int32"
    small = math.prod(ulead) == 1 and gm == urp
    spanned = uc == cols
    strips = not spanned and width > 2 * STRIP
    blocks = urp // gm
    # Ref order: operand, `count` updates, runtime start indices, output, then constant starts.
    indices = 0 if fixed else count * rank
    output = 1 + count + indices

    def kernel(refs, memory, result):
        """Kernel body: copies the operand to the output if needed, then applies each update."""
        sem = pl.scratch(pl.SemaphoreType.DMA)
        x = pallas.view(memory, refs[0], (math.prod(lead), rp, cp), dtype)
        out = pallas.view(result, refs[output], (math.prod(lead), rp, cp), dtype)
        slots = pl.scratch(pl.VMEM((SLOTS, bm, bn), dtype))
        flights = pl.scratch(pl.SemaphoreType.DMA((2, SLOTS)))
        buffers = depth if small else 2
        sources = pl.scratch(pl.VMEM((buffers, gm, ucp), dtype))
        windows = pl.scratch(pl.VMEM((buffers, gm, width), dtype))
        edge = pl.scratch(pl.VMEM((tile, width), dtype))
        loads, reads, stores = (pl.scratch(pl.SemaphoreType.DMA((buffers,))) for _ in range(3))

        def updated(m):
            """Update `m` viewed as (leading slabs, padded rows, padded cols)."""
            return pallas.view(memory, refs[1 + m], (math.prod(ulead), urp, ucp), dtype)

        def source(m, k, i, slot):
            """DMA of row block `i` of slab `k` of update `m` into source buffer `slot`."""
            rows_at = pl.multiple_of(i * gm, tile)
            return pl.make_async_copy(updated(m).at[k, pl.ds(rows_at, gm)], sources.at[slot], loads.at[slot])

        if small and not (resident and resident[6]):
            for j in range(min(depth, count)):
                source(_int(j), 0, _int(0), _int(j)).start()

        if copied:

            def moved(i, o):
                """Copies one input block to the output block."""
                o[...] = i[...]

            for first, length, step in spec["copied_rows"]:
                rows_of = pl.BlockSpec((None, step, cp), lambda l, i: (l, i, 0))
                within = (slice(None), pl.ds(first, length))
                pl.emit_pipeline(
                    moved, grid=(math.prod(lead), length // step), in_specs=[rows_of], out_specs=[rows_of]
                )(x.at[within], out.at[within])

        if fixed:

            def start(m, d):
                """Constant start of update `m` along dim `d`."""
                return refs[output + 1 + m * rank + d]

        else:
            held = pl.scratch(pl.SMEM((indices, SCALARS), "int32"))
            fetched = [
                pl.async_copy(pallas.view(memory, refs[1 + count + j], (SCALARS,), "int32"), held.at[j], sem)
                for j in range(indices)
            ]
            _waited(fetched)

            def start(m, d):
                """Runtime start of update `m` along dim `d`, clamped so the update fits."""
                return prims.min(prims.max(held[m * rank + d, 0], _int(0)), _int(shape[d] - update[d]))

        def one(m, waiting=False):
            """Applies update `m`, or with `waiting` waits for its pending store."""
            starts = [start(m, d) for d in range(rank)]
            # Window origin (a, b) is tile aligned; (dr, dc) is the update's offset inside it.
            r0, c0 = starts[-2], starts[-1]
            a = pl.multiple_of(r0 // tile * tile, tile)
            b = pl.multiple_of(prims.min(c0 // LANES * LANES, _int(cp - width)), LANES)
            dr, dc = r0 - a, c0 - b
            aligned = (dr == 0) & (c0 % LANES == 0)

            def slab(k):
                """Output slab index of the update's leading slab `k`."""
                at, rest = _int(0), k
                for d in reversed(range(len(ulead))):
                    at = at + (starts[d] + rest % ulead[d]) * math.prod(lead[d + 1 :])
                    rest = rest // ulead[d]
                return at

            def piece(k, t):
                """DMA of staged piece `t` of slab `k` of the update into its slot."""
                across = ucp // bn
                rows_at, cols_at = pl.multiple_of(t // across * bm, tile), pl.multiple_of(t % across * bn, LANES)
                return pl.make_async_copy(
                    updated(m).at[k, pl.ds(rows_at, bm), pl.ds(cols_at, bn)],
                    slots.at[t % SLOTS],
                    flights.at[0, t % SLOTS],
                )

            def staged(k, at):
                """Copies an aligned update slab through `SLOTS` VMEM pieces straight to `out`."""
                total = urp // bm * (ucp // bn)
                for t in range(min(SLOTS, total)):
                    piece(k, _int(t)).start()

                def store(t):
                    """DMA of staged piece `t` from its slot to its place in the output."""
                    across = ucp // bn
                    rows_at = pl.multiple_of(r0 + t // across * bm, tile)
                    cols_at = pl.multiple_of(c0 + t % across * bn, LANES)
                    to = out.at[at, pl.ds(rows_at, bm), pl.ds(cols_at, bn)]
                    return pl.make_async_copy(slots.at[t % SLOTS], to, flights.at[1, t % SLOTS])

                @pl.loop(0, total)
                def _(t):
                    """Waits for piece `t`, stores it, and reuses its slot for a later piece."""
                    piece(k, t).wait()
                    store(t).start()

                    @pl.when(t + SLOTS < total)
                    def _():
                        """Frees the slot of piece `t` and loads the piece that reuses it."""
                        store(t).wait()
                        piece(k, t + SLOTS).start()

                @pl.loop(max(0, total - SLOTS), total)
                def _(t):
                    """Waits for the last stores in flight."""
                    store(t).wait()

            def placed(tail, current, rows_of):
                """Rolls `current` (with the previous block's last tile rows) into window position."""
                v = prims.concatenate([tail, current], 0).to(wide)
                if width > ucp:
                    v = prims.concatenate([v, prims.full((tile + rows_of, width - ucp), 0, wide)], 1)
                v = pl.roll(v, dr, 0) if spanned else pl.roll(pl.roll(v, dr, 0), dc, 1)
                return prims.slice(v, (tile, 0), (tile + rows_of, width))

            def inside(top, rows_of):
                """Mask of window elements at rows from `top` that the update covers."""
                g = top + pl.broadcasted_iota("int32", (rows_of, width), 0)
                if spanned:
                    return (g >= r0) & (g < r0 + ur)
                c = pl.broadcasted_iota("int32", (rows_of, width), 1)
                return (g >= r0) & (g < r0 + ur) & (c >= dc) & (c < dc + uc)

            def window_reads(at, slot, top):
                """DMAs reading the output window at `top`: the full window, and what an interior block needs.

                An interior block needs nothing when the update spans all columns, and only edge strips for
                wide windows.
                """
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
                """Writes the update's rows that spill past the last window into the tile below it."""

                @pl.when(dr + ur > urp)
                def _():
                    """Read-modify-write of the tile of rows below the last window."""
                    top = pl.multiple_of(a + urp, tile)
                    target = out.at[at, pl.ds(top, tile), pl.ds(b, width)]
                    pl.async_copy(target, edge, sem).wait()
                    tail = sources[last, gm - tile : gm, :]
                    v = placed(tail, prims.full((tile, ucp), 0, dtype), tile)
                    edge[...] = prims.select(inside(top, tile), v, edge[...].to(wide)).to(dtype)
                    pl.async_copy(edge, target, sem).wait()

            def block(at, i, slot, tail_slot, loaded, prefetch):
                """Merges one source row block into its output window and returns the store DMA."""
                top = pl.multiple_of(a + i * gm, tile)
                # Blocks touching the update's top or bottom edge need the whole window read back.
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
                """Applies an unaligned update slab block by block, double-buffering loads and stores."""
                source(m, k, _int(0), _int(0)).start()

                def written(i):
                    """Store DMA of window block `i`."""
                    top = pl.multiple_of(a + i * gm, tile)
                    return pl.make_async_copy(
                        windows.at[i % 2], out.at[at, pl.ds(top, gm), pl.ds(b, width)], stores.at[i % 2]
                    )

                @pl.loop(0, blocks)
                def _(i):
                    """Writes block `i`, prefetching block `i + 1`."""

                    @pl.when(i >= 2)
                    def _():
                        """Waits for the store two blocks back before its buffer is reused."""
                        written(i - 2).wait()

                    def prefetch():
                        """Starts loading the next block."""

                        @pl.when(i + 1 < blocks)
                        def _():
                            """Starts the next block's load."""
                            source(m, k, i + 1, (i + 1) % 2).start()

                    block(at, i, i % 2, (i + 1) % 2, lambda: source(m, k, i, i % 2).wait(), prefetch).start()

                @pl.loop(max(0, blocks - 2), blocks)
                def _(i):
                    """Waits for the last stores in flight."""
                    written(i).wait()

                edged(at, (blocks - 1) % 2)

            if small:
                # Single-block updates: direct store when aligned, else one merged window.
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
                    """Merges the update through one window, then patches the tile below."""
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
                """Applies slab `k` of the update."""
                at = slab(k)
                if not whole:
                    streamed(k, at)
                    return
                pl.when(aligned)(lambda: staged(k, at))
                pl.when(~aligned)(lambda: streamed(k, at))

        def refill(j):
            """Starts loading the update `depth` ahead into slot `j`'s buffer."""

            @pl.when(j + depth < count)
            def _():
                """Starts the load when that update exists."""
                source(j + depth, 0, _int(0), j % depth).start()

        if resident:
            # Resident: merge every update into a VMEM copy of the box they touch, then write it once.
            top, bottom, left, right, at, covered, direct = resident
            height, across = bottom - top, right - left
            tall, wide_as = min(urp + tile, height), min(width, across)
            kept_box = pl.scratch(pl.VMEM((height, across), dtype))
            box = out.at[at, pl.ds(top, height), pl.ds(left, across)]
            if not covered:
                pl.async_copy(box, kept_box, sem).wait()

            def placed_directly(m):
                """DMA of aligned update `m` straight into the resident box."""
                rows_at = pl.multiple_of(start(m, rank - 2) - top, tile)
                cols_at = pl.multiple_of(start(m, rank - 1) - left, LANES)
                return pl.make_async_copy(
                    updated(m).at[0], kept_box.at[pl.ds(rows_at, urp), pl.ds(cols_at, ucp)], loads.at[0]
                )

            if direct and lag == 0:

                @pl.loop(0, count)
                def _(m):
                    """Places update `m` and waits before the next."""
                    placed_directly(m).start()
                    placed_directly(m).wait()

            elif direct:

                @pl.loop(0, count)
                def _(m):
                    """Starts placing update `m`."""
                    placed_directly(m).start()

                @pl.loop(0, count)
                def _(m):
                    """Waits for update `m`'s placement."""
                    placed_directly(m).wait()

            if direct:

                pl.async_copy(kept_box, box, sem).wait()
                return

            def kept(m):
                """Merges update `m` into the resident box."""
                r0, c0 = start(m, rank - 2) - top, start(m, rank - 1) - left
                ra = pl.multiple_of(prims.min(r0 // tile * tile, _int(height - tall)), tile)
                cb = pl.multiple_of(prims.min(c0 // LANES * LANES, _int(across - wide_as)), LANES)
                dr, dc = r0 - ra, c0 - cb
                slot = m % depth
                source(m, 0, _int(0), slot).wait()

                def fast():
                    """Copies the aligned update into the box."""
                    rows_at, cols_at = pl.multiple_of(r0, tile), pl.multiple_of(c0, LANES)
                    kept_box[pl.ds(rows_at, gm), pl.ds(cols_at, ucp)] = sources[slot, :, :]

                def slow():
                    """Rolls the update into place and selects it over the box contents."""
                    v = sources[slot, :, :].to(wide)
                    if tall > gm:
                        v = prims.concatenate([v, prims.full((tall - gm, ucp), 0, wide)], 0)
                    if wide_as > ucp:
                        v = prims.concatenate([v, prims.full((tall, wide_as - ucp), 0, wide)], 1)
                    v = pl.roll(v, dr, 0) if spanned else pl.roll(pl.roll(v, dr, 0), dc, 1)
                    g = pl.broadcasted_iota("int32", (tall, wide_as), 0)
                    c = pl.broadcasted_iota("int32", (tall, wide_as), 1)
                    inside = (g >= dr) & (g < dr + ur)
                    if not spanned:
                        inside = inside & (c >= dc) & (c < dc + uc)
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
                """Merges update `m`."""
                kept(m)

            pl.async_copy(kept_box, box, sem).wait()
            return

        @pl.loop(0, count)
        def _(m):
            """Applies update `m`, waiting on the previous (or this) one's store before reusing its buffer."""
            one(m)
            if not small:
                return
            if lag == 0:
                one(m, waiting=True)
                refill(m)
                return

            @pl.when(m >= 1)
            def _():
                """Waits for the previous update's store and refills its buffer."""
                one(m - 1, waiting=True)
                refill(m - 1)

        if small and lag:
            one(_int(count - 1), waiting=True)

    return pallas.program(kernel, spec)
