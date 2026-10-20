from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.elementwise.tpu import computed
from lumen.ops.tpu import pallas


def program(spec):
    """The dot of ``spec``: ``batch``, ``lead``, ``m``, ``n``, ``k``,
    ``transposed`` (a ``[n, k]`` rhs), the ``operands`` and ``value`` (its
    result's) dtypes, ``blocks`` (bm, bn, bk), ``widths`` (its members:
    column ranges, in order), ``inputs`` (its epilogue's other operands,
    each ``[dtype, 0-d, member]``), ``nodes`` (its epilogue, as the
    elementwise kernel's, its values the result, the inputs, then the
    nodes) and ``outputs`` (each ``[value, member, dtype]``). Its scalars
    are the rows of the lhs, the rhs, the inputs and the outputs."""
    batch, lead, m, n, k = spec["batch"], spec["lead"], spec["m"], spec["n"], spec["k"]
    bm, bn, bk = spec["blocks"]
    t = spec["transposed"]
    dtype, value = spec["operands"], spec["value"]
    mb, nb, kb = m // bm, n // bn, k // bk
    widths, inputs, nodes, outputs = spec["widths"], spec["inputs"], spec["nodes"], spec["outputs"]
    starts = [sum(widths[:p]) // bn for p in range(len(widths))]
    counts = [w // bn for w in widths]
    blocked = [i for i, (_, scalar, _) in enumerate(inputs) if not scalar]

    def rhs_batch(b):
        if lead == 1:
            return b
        if isinstance(b, int):
            return b // lead
        return prims.div(b, prims.full((), lead, "int32"))

    def needed(indices):
        """The values ``indices`` are computed from, and they."""
        first = 1 + len(inputs)
        stack, seen = list(indices), set()
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                if v >= first:
                    stack.extend(nodes[v - first]["args"])
        return seen

    def kernel(rows, memory, result, *scratch):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        x = pallas.view(memory, rows[0], (batch * lead, m, k), dtype)
        y = pallas.view(memory, rows[1], (batch, n, k) if t else (batch, k, n), dtype)
        scalars = {}
        for i, (d, scalar, _) in enumerate(inputs):
            if scalar:
                scalars[i] = pl.scratch(pl.VMEM((1,), d))
                copy(pallas.view(memory, rows[2 + i], (), d), scalars[i])
        read = {i: s[...] for i, s in scalars.items()}
        extra = [
            pallas.view(memory, rows[2 + i], (batch * lead, m, widths[inputs[i][2]]), inputs[i][0]) for i in blocked
        ]
        base = 2 + len(inputs)
        outs = [
            pallas.view(result, rows[base + o], (batch * lead, m, widths[p]), d) for o, (_, p, d) in enumerate(outputs)
        ]
        x_spec = pl.BlockSpec((None, bm, bk), lambda b, i, j, s: (b, i, s))
        if t:
            y_spec = pl.BlockSpec((None, bn, bk), lambda b, i, j, s: (rhs_batch(b), j, s))
        else:
            y_spec = pl.BlockSpec((None, bk, bn), lambda b, i, j, s: (rhs_batch(b), s, j))

        def column(p):
            start, count = starts[p], counts[p]

            def index(b, i, j, s):
                if isinstance(j, int):
                    return b, i, min(max(j - start, 0), count - 1)
                shifted = prims.max(prims.sub(j, prims.full((), start, "int32")), prims.full((), 0, "int32"))
                return b, i, prims.neg(prims.max(prims.neg(shifted), prims.full((), 1 - count, "int32")))

            return pl.BlockSpec((None, bm, bn), index)

        e_specs = [column(inputs[i][2]) for i in blocked]
        o_specs = [column(p) for _, p, _ in outputs]

        def finish(acc, blocks, obs):
            result = acc.to(value)
            members = sorted({p for _, p, _ in outputs})
            for p in members:
                written = [o for o, (_, q, _) in enumerate(outputs) if q == p]

                def store(written=written):
                    wanted = needed([outputs[o][0] for o in written])
                    values = [result] + [
                        read[i] if i in read else blocks[blocked.index(i)][...] for i in range(len(inputs))
                    ]
                    first = len(values)
                    values += [None] * len(nodes)
                    for v in sorted(w for w in wanted if w >= first):
                        node = nodes[v - first]
                        values[v] = computed(
                            [{**node, "args": list(range(len(node["args"])))}],
                            [values[a] for a in node["args"]],
                            (bm, bn),
                        )[-1]
                    for o in written:
                        obs[o][...] = values[outputs[o][0]].to(outputs[o][2])

                if len(members) == 1:
                    store()
                    continue
                j = pl.program_id(2)
                pl.when((j >= starts[p]) & (j < starts[p] + counts[p]))(store)

        def body(xb, yb, *rest):
            blocks, obs = rest[: len(blocked)], rest[len(blocked) :]
            if kb == 1:
                finish(pl.dot(xb[...], yb[...], trans_b=t), blocks, obs)
                return
            step, acc = pl.program_id(3), scratch[0]

            @pl.when(step == 0)
            def _():
                acc[...] = 0.0

            acc[...] += pl.dot(xb[...], yb[...], trans_b=t)

            @pl.when(step == kb - 1)
            def _():
                finish(acc[...], blocks, obs)

        grid = (batch * lead, mb, nb, kb)
        pl.emit_pipeline(body, grid=grid, in_specs=[x_spec, y_spec] + e_specs, out_specs=o_specs)(x, y, *extra, *outs)

    return pallas.program(kernel, spec, scratch=[pl.VMEM((bm, bn), "float32")] if kb > 1 else [])
