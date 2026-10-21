import lumen.functional as F
from lumen import pallas as pl
from lumen.codegen.tpu import computed
from lumen.ops.tpu import pallas


def program(spec):
    """The dot of ``spec``: ``batch``, ``lead``, ``m``, ``n``, ``k``,
    ``transposed`` (a ``[n, k]`` rhs), ``lhs_transposed`` (a ``[k, m]``
    lhs), ``lhs`` (the sizes of the lhs's parts, buffers of their own side
    by side along k if ``lhs_k``, else m; none: one buffer), the
    ``operands`` and ``value`` (its result's) dtypes, ``blocks`` (bm, bn,
    bk), ``widths`` (its members: ranges of its rows if ``split_rows``, else of
    its columns, in order), ``inputs`` (its epilogue's other operands,
    each ``[dtype, 0-d, member]``), ``nodes`` (its epilogue, as the
    elementwise kernel's, its values the result, the inputs, then the
    nodes) and ``outputs`` (each ``[value, member, dtype, n, start]``:
    written at columns ``start`` on of a buffer ``n`` wide, or, ``n`` 0, a
    buffer of its own). With ``together``, members of one width, each step
    computes every member's block at its columns, from a stream of the rhs's
    blocks each, so the epilogue reads them all. Its scalars are the rows
    of the lhs's parts, the rhs, the inputs and the outputs."""
    batch, lead, m, n, k = spec["batch"], spec["lead"], spec["m"], spec["n"], spec["k"]
    bm, bn, bk = spec["blocks"]
    t, tl = spec["transposed"], spec["lhs_transposed"]
    dtype, value = spec["operands"], spec["value"]
    mb, nb, kb = m // bm, n // bn, k // bk
    widths, inputs, nodes, outputs = spec["widths"], spec["inputs"], spec["nodes"], spec["outputs"]
    rows_split, lhs_k, together = spec["split_rows"], spec["lhs_k"], spec["together"]
    parts = spec["lhs"] or [k if lhs_k else m]
    unit = bm if rows_split else bn
    starts = [sum(widths[:p]) // unit for p in range(len(widths))]
    counts = [w // unit for w in widths]
    part_unit = bk if lhs_k else bm
    part_starts = [sum(parts[:p]) // part_unit for p in range(len(parts))]
    part_counts = [w // part_unit for w in parts]
    blocked = [i for i, (_, scalar, _) in enumerate(inputs) if not scalar]
    split = len(parts) > 1
    results = len(widths)
    streams = results if together else 1
    if together:
        nb = widths[0] // bn

    def rhs_batch(b):
        if lead == 1:
            return b
        return b // lead

    def clamp(x, start, count):
        if isinstance(x, int):
            return min(max(x - start, 0), count - 1)
        return F.minimum(F.maximum(x - start, 0), count - 1)

    def within(x, start, count):
        return (x >= start) & (x < start + count)

    def product(xb, yb):
        if not tl:
            return pl.dot(xb[...], yb[...], trans_b=t)
        numbers = (([0], [1 if t else 0]), ([], []))
        return pl.dot_general(xb[...], yb[...], numbers, preferred_element_type="float32")

    def needed(indices):
        """The values ``indices`` are computed from, and they."""
        first = results + len(inputs)
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

        def lhs_shape(size):
            free, contracting = (m, size) if lhs_k else (size, k)
            return (batch * lead, contracting, free) if tl else (batch * lead, free, contracting)

        xs = [pallas.view(memory, rows[p], lhs_shape(size), dtype) for p, size in enumerate(parts)]
        r = len(parts)
        y = pallas.view(memory, rows[r], (batch, n, k) if t else (batch, k, n), dtype)
        scalars = {}
        for i, (d, scalar, _) in enumerate(inputs):
            if scalar:
                scalars[i] = pl.scratch(pl.VMEM((1,), d))
                copy(pallas.view(memory, rows[r + 1 + i], (), d), scalars[i])
        read = {i: s[...] for i, s in scalars.items()}

        def member_shape(p):
            return (batch * lead, widths[p], n) if rows_split else (batch * lead, m, widths[p])

        extra = [pallas.view(memory, rows[r + 1 + i], member_shape(inputs[i][2]), inputs[i][0]) for i in blocked]
        base = r + 1 + len(inputs)
        outs = [
            pallas.view(result, rows[base + o], (batch * lead, m, whole) if whole else member_shape(p), d)
            for o, (_, p, d, whole, _) in enumerate(outputs)
        ]

        def lhs_spec(p):
            start, count = part_starts[p], part_counts[p]

            def index(b, i, j, s):
                if split and lhs_k:
                    s = clamp(s, start, count)
                elif split:
                    i = clamp(i, start, count)
                return (b, s, i) if tl else (b, i, s)

            return pl.BlockSpec((None, bk, bm) if tl else (None, bm, bk), index)

        x_specs = [lhs_spec(p) for p in range(len(parts))]

        def rhs_spec(p):
            at = starts[p] if together else 0

            def index(b, i, j, s):
                if at:
                    j = j + at
                return (rhs_batch(b), j, s) if t else (rhs_batch(b), s, j)

            return pl.BlockSpec((None, bn, bk) if t else (None, bk, bn), index)

        y_specs = [rhs_spec(p) for p in range(streams)]

        def member_spec(p, at=0):
            start, count = starts[p], counts[p]

            def index(b, i, j, s):
                if rows_split:
                    return b, clamp(i, start, count), j
                column = j if together else clamp(j, start, count)
                if at and isinstance(column, int):
                    column += at // bn
                elif at:
                    column = column + at // bn
                return b, i, column

            return pl.BlockSpec((None, bm, bn), index)

        e_specs = [member_spec(inputs[i][2]) for i in blocked]
        o_specs = [member_spec(p, at) for _, p, _, _, at in outputs]

        def finish(accs, blocks, obs):
            products = [a.to(value) for a in accs] * (results // len(accs))
            members = [0] if together else sorted({o[1] for o in outputs})
            for p in members:
                written = [o for o, out in enumerate(outputs) if together or out[1] == p]

                def store(written=written):
                    wanted = needed([outputs[o][0] for o in written])
                    values = products + [
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
                at = pl.program_id(1 if rows_split else 2)
                pl.when(within(at, starts[p], counts[p]))(store)

        def body(*refs):
            xbs, ybs = refs[: len(parts)], refs[len(parts) : len(parts) + streams]
            rest = refs[len(parts) + streams :]
            blocks, obs = rest[: len(blocked)], rest[len(blocked) :]
            if kb == 1 and not split:
                finish([product(xbs[0], yb) for yb in ybs], blocks, obs)
                return
            step = pl.program_id(3)

            @pl.when(step == 0)
            def _():
                for acc in scratch:
                    acc[...] = 0.0

            if not split:
                for acc, yb in zip(scratch, ybs):
                    acc[...] += product(xbs[0], yb)
            for p, xb in enumerate(xbs if split else []):
                at = pl.program_id(3 if lhs_k else 1)

                @pl.when(within(at, part_starts[p], part_counts[p]))
                def _(xb=xb):
                    scratch[0][...] += product(xb, ybs[0])

            @pl.when(step == kb - 1)
            def _():
                finish([acc[...] for acc in scratch], blocks, obs)

        grid = (batch * lead, mb, nb, kb)
        in_specs = x_specs + y_specs + e_specs
        pl.emit_pipeline(body, grid=grid, in_specs=in_specs, out_specs=o_specs)(*xs, *[y] * streams, *extra, *outs)

    accumulated = kb > 1 or split
    scratch = [pl.VMEM((bm, bn), "float32") for _ in range(streams)] if accumulated else []
    return pallas.program(kernel, spec, scratch=scratch)
