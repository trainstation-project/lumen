import math

import lumen
import lumen.functional as F
from lumen import pallas as pl
from lumen.codegen.tpu import computed
from lumen.kernels.tpu import pallas


def _identity(name, dtype, shape):
    if name == "sum":
        return lumen.full(shape, 0, dtype)
    if dtype in ("float32", "bfloat16", "float16"):
        return lumen.full(shape, float("-inf"), dtype)
    if dtype.startswith("uint"):
        return lumen.full(shape, 0, dtype)
    return lumen.full(shape, -(2 ** (8 * {"int8": 1, "int16": 2}.get(dtype, 4) - 1)), dtype)


def _exact_max(x, axes, dtype):
    """``x``'s 32-bit integers' maximum over ``axes`` through float32
    maxima of their 16-bit halves (each exact in float32): the high
    halves' maximum, then the low halves' among the values holding it."""
    high, low = x >> 16, x & 0xFFFF
    top = F.amax(high.to("float32"), axes)
    at_top = high.to("float32") == top
    rest = F.amax(F.where(at_top, low.to("float32"), -1.0), axes)
    return (top.to(dtype) << 16) | rest.to(dtype)


def _schedule(spec):
    """The views, blocks and grid of ``spec``'s layout: ``rows`` (the last
    axis: ``[m, n]`` to ``[m]``, ``[lead, h, m, n]`` to ``[lead, h, m]``),
    ``columns`` (every axis but the last: ``[lead, m, n]`` to ``[n]``) or
    ``total`` (every axis, to a scalar); the grid's reduced axes last."""
    layout, blocks = spec["layout"], spec["blocks"]
    lead, m, n = spec["lead"], spec["m"], spec["n"]
    if layout == "rows" and spec["rank"] == 2:
        _, bm, bn = blocks
        return dict(
            x=((m, n), pl.BlockSpec((bm, bn), lambda i, s: (i, s)), (bm, bn)),
            o=((m,), pl.BlockSpec((bm,), lambda i, s: (i,)), (bm,)),
            grid=(m // bm, n // bn),
            axes=(1,),
            reduced=1,
        )
    if layout == "rows":
        bh, bm, bn = blocks
        h = spec["h"]
        return dict(
            x=(
                (lead, h, m, n),
                pl.BlockSpec((None, bh, bm, bn), lambda l, i, j, s: (l, i, j, s)),
                (bh, bm, bn),
            ),
            o=((lead, h, m), pl.BlockSpec((None, bh, bm), lambda l, i, j, s: (l, i, j)), (bh, bm)),
            grid=(lead, h // bh, m // bm, n // bn),
            axes=(2,),
            reduced=1,
        )
    if layout == "columns":
        bm, bn = blocks
        return dict(
            x=((lead, m, n), pl.BlockSpec((None, bm, bn), lambda j, l, i: (l, i, j)), (bm, bn)),
            o=((n,), pl.BlockSpec((bn,), lambda j, l, i: (j,)), (bn,)),
            grid=(n // bn, lead, m // bm),
            axes=(0,),
            reduced=2,
        )
    if spec["flat"]:
        (b,) = blocks
        x = ((n,), pl.BlockSpec((b,), lambda s: (s,)), (b,))
        return dict(x=x, o=((), None, (1,)), grid=(n // b,), axes=(0,), reduced=1)
    bm, bn = blocks
    return dict(
        x=((lead, m, n), pl.BlockSpec((None, bm, bn), lambda l, i, s: (l, i, s)), (bm, bn)),
        o=((), None, (1,)),
        grid=(lead, m // bm, n // bn),
        axes=(0, 1),
        reduced=3,
    )


def program(spec):
    """A reduction with its prologue and epilogue (the hand-written
    schedule; the fused work generated: ``lumen/codegen/tpu/reduction.rs``).
    ``inputs`` (each ``[dtype, kind]``: ``in``, the reduced operand's shape;
    ``out``, the result's; or ``scalar``), ``prologue`` (nodes computing the
    reduced ``operand`` on its blocks), ``reduce`` (``sum`` in ``accum``, or
    ``max``), ``epilogue`` (nodes after it, the reduced value numbered after
    the prologue's) and ``result`` (the value written, ``output``). A step
    reduces an operand block into a VMEM accumulator; the last step of a
    block of results runs the epilogue. Its scalars are the operands' rows,
    then the output's."""
    s = _schedule(spec)
    (x_view, x_spec, x_block), (o_view, o_spec, o_block) = s["x"], s["o"]
    grid, reduced = s["grid"], s["reduced"]
    inputs, accum, name = spec["inputs"], spec["accum"], spec["reduce"]["name"]
    total = o_spec is None
    steps = math.prod(grid[len(grid) - reduced :])

    def reduce(x):
        if name == "sum":
            r = F.sum(x.to(accum), s["axes"])
        elif total and accum in ("int32", "uint32"):
            r = _exact_max(x, s["axes"], accum)
        elif total and accum != "float32":
            r = F.amax(x.to("float32"), s["axes"]).to(accum)
        else:
            r = F.amax(x, s["axes"])
        return r.expand((1,)) if total else r

    def combine(a, b):
        return a + b if name == "sum" else F.maximum(a, b)

    def kernel(rows, memory, result, acc):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        read = {}
        for i, (dtype, kind) in enumerate(inputs):
            if kind == "scalar" or (kind == "out" and total):
                v = pl.scratch(pl.VMEM((1,), dtype))
                copy(pallas.view(memory, rows[i], (), dtype), v)
                read[i] = v[...]
        ins = [i for i, (_, kind) in enumerate(inputs) if kind == "in" and i not in read]
        outs = [i for i, (_, kind) in enumerate(inputs) if kind == "out" and i not in read]
        piped = [pallas.view(memory, rows[i], x_view, inputs[i][0]) for i in ins]
        piped += [pallas.view(memory, rows[i], o_view, inputs[i][0]) for i in outs]
        specs = [x_spec] * len(ins) + [o_spec] * len(outs)
        out = pallas.view(result, rows[len(inputs)], o_view, spec["output"])

        def finish(r, given):
            values = [read.get(i, given.get(i)) for i in range(len(inputs))]
            values += [None] * len(spec["prologue"]) + [r]
            return computed(spec["epilogue"], values, o_block)[spec["result"]]

        def body(*refs):
            blocks = dict(zip(ins + outs, refs))
            values = [read[i] if i in read else blocks[i][...] if i in ins else None for i in range(len(inputs))]
            partial = reduce(computed(spec["prologue"], values, x_block)[spec["operand"]])
            if steps == 1 and not total:
                refs[-1][...] = finish(partial, {i: blocks[i][...] for i in outs})
                return
            ids = [pl.program_id(a) for a in range(len(grid) - reduced, len(grid))]
            first = last = None
            for a, k in zip(ids, grid[len(grid) - reduced :]):
                first = (a == 0) if first is None else first & (a == 0)
                last = (a == k - 1) if last is None else last & (a == k - 1)

            @pl.when(first)
            def _():
                acc[...] = _identity(name, accum, o_block)

            acc[...] = combine(acc[...], partial)
            if not total:

                @pl.when(last)
                def _():
                    refs[-1][...] = finish(acc[...], {i: blocks[i][...] for i in outs})

        if total:
            pl.emit_pipeline(body, grid=grid, in_specs=specs, out_specs=[])(*piped)
            vmem = pl.scratch(pl.VMEM((1,), spec["output"]))
            vmem[...] = finish(acc[...], {})
            copy(vmem, out)
            return
        pl.emit_pipeline(body, grid=grid, in_specs=specs, out_specs=[o_spec])(*piped, out)

    return pallas.program(kernel, spec, scratch=[pl.VMEM(o_block, accum)])
