"""Pallas TPU kernel for row fusions: elementwise and per-row work over blocks of whole rows."""

from lumen.codegen.devices.tpu import pallas, per_row
from lumen.graph.tracing import prims
from lumen.kernels import pallas as pl

# 32-bit words a TPU tile holds per lane: 8 rows of a 32-bit table, 16 of a 16-bit one in pairs.
SUBLANES = 8


def _gathered(gather, table, ids, tiles, picked, sems, n, bm):
    """The block's ``bm`` rows of ``table`` at ``ids`` (in SMEM), this pipeline step's, moved as bits (exact, subnormals
    and ``-0.0`` kept): each row's TPU tile fetched into ``tiles`` as 32-bit words, its word picked by a sum over the
    tile's words with the others masked to 0, a packed 16-bit row's half shifted out and paired with its neighbor's into
    ``picked``'s words, read back as the table's dtype."""
    pack = 2 if gather["dtype"] == "bfloat16" else 1
    base = pl.program_id(0) * bm
    rows = [ids[(base + k,) if gather["flat"] else (0, base + k)] for k in range(bm)]

    copies = []
    for k, row in enumerate(rows):
        start = pl.multiple_of(row // pack // SUBLANES * SUBLANES, SUBLANES)
        c = pl.make_async_copy(table.at[pl.ds(start, SUBLANES)], tiles.at[k], sems.at[k])
        c.start()
        copies.append(c)

    shape = (SUBLANES, n)
    sublane = pl.broadcasted_iota("int32", shape, 0)
    bits = []
    for k, (c, row) in enumerate(zip(copies, rows)):
        c.wait()
        mine = prims.eq(sublane, prims.broadcast_in_dim(row // pack % SUBLANES, shape, ()))
        word = prims.reduce_sum(prims.select(mine, tiles[k], prims.full(shape, 0, "int32")), (0,), "int32")
        if pack > 1:
            half = prims.broadcast_in_dim(row % pack * 16, (n,), ())
            word = pl.bitwise_and(pl.shift_right_logical(word, half), prims.full((n,), 0xFFFF, "int32"))
        bits.append(word)

    # Two 16-bit rows share a word, the even row in its low half.
    for w in range(-(-bm // pack)):
        word = bits[w * pack]
        if pack > 1 and w * pack + 1 < bm:
            word = pl.bitwise_or(word, pl.shift_left(bits[w * pack + 1], prims.full((n,), 16, "int32")))
        picked[pl.ds(w, 1), :] = prims.reshape(word, (1, n))
    words = picked if gather["dtype"] == "int32" else picked.bitcast(gather["dtype"])
    return words[pl.ds(0, bm), :]


def program(spec):
    """A row fusion's kernel: ``height`` rows of ``n``, ``block`` rows a step,
    whole rows in VMEM. ``inputs`` (each ``[dtype, kind]``), ``nodes`` (as
    an elementwise kernel's, each with the ``kind`` of its value) and
    ``outputs`` (each ``[value, dtype, kind]``); a kind is ``full`` (a block
    of rows), ``row`` (a value a row), ``lanes`` (a value a column, the same
    for every row) or ``scalar``. Its scalars are the operands' rows, then
    the outputs'. With ``gather`` (``vocab``, ``dtype``: float32,
    bfloat16 or int32, ``flat``: ids of one dimension), the first input is rows of a ``[vocab, n]``
    table at int32 ids, the kernel's first two operands, which the kernel
    fetches itself; the other inputs follow them."""
    rows, n, bm = spec["height"], spec["n"], spec["block"]
    shapes = {"full": (bm, n), "row": (bm, 1), "lanes": (1, n), "scalar": (1, 1)}
    views = {"full": (rows, n), "row": (rows, 1), "lanes": (n,)}
    blocks = {
        "full": pl.BlockSpec((bm, n), lambda i: (i, 0)),
        "row": pl.BlockSpec((bm, 1), lambda i: (i, 0)),
        "lanes": pl.BlockSpec((n,), lambda i: (0,)),
    }
    inputs, outputs, gather = spec["inputs"], spec["outputs"], spec.get("gather")
    # A gathered first input takes the table's and the ids' operands, so the rest start one later.
    shift = 1 if gather else 0

    def kernel(refs, memory, result):
        """Pallas kernel body: reads scalar operands, then pipelines row blocks through the fused nodes."""

        def copy(src, dst):
            """Copies `src` to `dst` with a blocking DMA."""
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        fetch = None
        if gather:
            pack = 2 if gather["dtype"] == "bfloat16" else 1
            # The table as 32-bit words, its rows padded to whole tiles as its buffer holds them.
            words = -(-gather["vocab"] // pack)
            table = pallas.view(memory, refs[0], (-(-words // SUBLANES) * SUBLANES, n), "int32")
            # Ids of one dimension are stored flat, others as a tiled `[1, rows]` row.
            at = (rows,) if gather["flat"] else (1, rows)
            ids = pl.scratch(pl.SMEM(at, "int32"))
            copy(pallas.view(memory, refs[1], at, "int32"), ids)
            tiles = pl.scratch(pl.VMEM((bm, SUBLANES, n), "int32"))
            # The block's words, whole tiles of them, read back through a view in the table's dtype.
            picked = pl.scratch(pl.VMEM((-(-(-(-bm // pack)) // SUBLANES) * SUBLANES, n), "int32"))
            sems = pl.scratch(pl.SemaphoreType.DMA((bm,)))

            def fetch():
                """This step's gathered rows."""
                return _gathered(gather, table, ids, tiles, picked, sems, n, bm)

        piped, specs, scalars = [], [], {}
        for i, (dtype, kind) in enumerate(inputs):
            if gather and i == 0:
                continue
            ref = refs[i + shift]
            if kind == "scalar":
                scalars[i] = pl.scratch(pl.VMEM((1,), dtype))
                copy(pallas.view(memory, ref, (), dtype), scalars[i])
                continue
            piped.append(pallas.view(memory, ref, views[kind], dtype))
            specs.append(blocks[kind])
        read = {i: s[...].expand((1, 1)) for i, s in scalars.items()}
        outs = [
            pallas.view(result, refs[len(inputs) + shift + k], views[kind], dtype)
            for k, (_, dtype, kind) in enumerate(outputs)
        ]

        def body(*refs_):
            """Pipeline step: computes the fused nodes on one block of rows and writes each output block."""
            given = iter(refs_[: len(piped)])
            values = []
            for i, (_, kind) in enumerate(inputs):
                if gather and i == 0:
                    values.append(fetch())
                    continue
                if i in read:
                    values.append(read[i])
                    continue
                v = next(given)[...]
                # per-column operands arrive as `(n,)` blocks and broadcast against rows as `(1, n)`
                values.append(v.reshape(1, n) if kind == "lanes" else v)
            values = per_row(spec["nodes"], values, shapes)
            for ref, (k, _, _) in zip(refs_[len(piped) :], outputs):
                ref[...] = values[k]

        o_specs = [blocks[kind] for _, _, kind in outputs]
        pl.emit_pipeline(body, grid=(rows // bm,), in_specs=specs, out_specs=o_specs)(*piped, *outs)

    return pallas.program(kernel, spec)
