import math

from lumen import pallas as pl
from lumen.graph import prims
from lumen.kernels.tpu import pallas

LANES = 128

SCALARS = 128


def _up(n, m):
    return -(-n // m) * m


def _int(n):
    return prims.full((), n, "int32")


def program(spec):
    shape, update, dtype, tile = spec["shape"], spec["update"], spec["dtype"], spec["tile"]
    bm, bn, copied = spec["bm"], spec["bn"], spec["copy"]
    rank = len(shape)
    (rows, cols), (ur, uc) = shape[-2:], update[-2:]
    rp, cp, urp, ucp = _up(rows, tile), _up(cols, LANES), _up(ur, tile), _up(uc, LANES)
    lead, ulead = shape[:-2], update[:-2]
    wr, wc = min(bm + tile, rp), min(bn + LANES, cp)
    whole = (ur % tile == 0 or ur == rows) and (uc % LANES == 0 or uc == cols)
    wide = "float32" if dtype.startswith("bfloat") or dtype.startswith("float") else "int32"

    single = math.prod(ulead) == 1 and urp == bm and ucp == bn

    def kernel(refs, memory, result):
        sem = pl.scratch(pl.SemaphoreType.DMA)
        sems = pl.scratch(pl.SemaphoreType.DMA((2,)))
        x = pallas.view(memory, refs[0], (math.prod(lead), rp, cp), dtype)
        u = pallas.view(memory, refs[1], (math.prod(ulead), urp, ucp), dtype)
        out = pallas.view(result, refs[2 + rank], (math.prod(lead), rp, cp), dtype)
        window = pl.scratch(pl.VMEM((wr, wc), dtype))
        block = pl.scratch(pl.VMEM((wr, wc), dtype))
        staged = block.at[pl.ds(0, bm), pl.ds(0, bn)]

        def piece(k, i, j):
            a, b = pl.multiple_of(i * bm, tile), pl.multiple_of(j * bn, LANES)
            return pl.make_async_copy(u.at[k, pl.ds(a, bm), pl.ds(b, bn)], staged, sems.at[1])

        early = piece(0, 0, 0) if single else None
        if single:
            early.start()

        def loaded(k, i, j):
            if single:
                early.wait()
                return
            c = piece(k, i, j)
            c.start()
            c.wait()

        if copied:
            pl.async_copy(x, out, sem).wait()
        held = pl.scratch(pl.SMEM((rank, SCALARS), "int32"))
        fetched = [pl.async_copy(pallas.view(memory, refs[2 + d], (SCALARS,), "int32"), held.at[d], sem) for d in range(rank)]
        for c in fetched:
            c.wait()
        starts = [prims.min(prims.max(held[d, 0], _int(0)), _int(shape[d] - update[d])) for d in range(rank)]
        r0, c0 = starts[-2], starts[-1]
        row = pl.broadcasted_iota("int32", (wr, wc), 0)
        col = pl.broadcasted_iota("int32", (wr, wc), 1)

        def slab(k):
            at, rest = _int(0), k
            for d in reversed(range(len(ulead))):
                at = at + (starts[d] + rest % ulead[d]) * math.prod(lead[d + 1 :])
                rest = rest // ulead[d]
            return at

        def merged(k, at, i, j):
            r, c = r0 + i * bm, c0 + j * bn
            a = pl.multiple_of(prims.min(r // tile * tile, _int(rp - wr)), tile)
            b = pl.multiple_of(prims.min(c // LANES * LANES, _int(cp - wc)), LANES)
            dr, dc = r - a, c - b
            cr, cc = prims.min(_int(bm), ur - i * bm), prims.min(_int(bn), uc - j * bn)
            target = out.at[at, pl.ds(a, wr), pl.ds(b, wc)]
            read = pl.async_copy(target, window, sems.at[0])
            loaded(k, i, j)
            read.wait()
            moved = pl.roll(pl.roll(block[...].to(wide), dr, 0), dc, 1)
            inside = (row >= dr) & (row < dr + cr) & (col >= dc) & (col < dc + cc)
            window[...] = prims.select(inside, moved, window[...].to(wide)).to(dtype)
            pl.async_copy(window, target, sem).wait()

        @pl.loop(0, math.prod(ulead))
        def _(k):
            at = slab(k)

            def blocks():
                @pl.loop(0, urp // bm)
                def _(i):
                    @pl.loop(0, ucp // bn)
                    def _(j):
                        merged(k, at, i, j)

            if not whole:
                blocks()
                return
            aligned = (r0 % tile == 0) & (c0 % LANES == 0)

            @pl.when(aligned)
            def _():
                @pl.loop(0, urp // bm)
                def _(i):
                    @pl.loop(0, ucp // bn)
                    def _(j):
                        loaded(k, i, j)
                        a, b = i * bm, j * bn
                        to = out.at[at, pl.ds(pl.multiple_of(r0 + a, tile), bm), pl.ds(pl.multiple_of(c0 + b, LANES), bn)]
                        pl.async_copy(staged, to, sem).wait()

            pl.when(~aligned)(blocks)

    return pallas.program(kernel, spec)
