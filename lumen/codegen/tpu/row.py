from lumen import pallas as pl
from lumen.graph import prims
from lumen.ops.tpu import pallas


def replayed(nodes, values, shapes):
    values = list(values)
    for node in nodes:
        args = [values[i] for i in node["args"]]
        name, params, shape = node["name"], node["params"], shapes[node["kind"]]
        if name in ("reduce_sum", "reduce_max"):
            if name == "reduce_sum":
                reduced = prims.reduce_sum(args[0], (1,), params["accum_dtype"])
            else:
                reduced = prims.reduce_max(args[0], (1,))
            values.append(prims.reshape(reduced, shape))
        elif name in ("broadcast_in_dim", "reshape"):
            x = args[0]
            values.append(x if tuple(x.shape) == shape else prims.broadcast_in_dim(x, shape, (0, 1)))
        elif name == "full":
            values.append(prims.full(shape, params["fill_value"], params["dtype"]))
        else:
            values.append(prims.bind(name, *args, **params))
    return values


def program(spec):
    """A row fusion's kernel: ``height`` rows of ``n``, ``block`` rows a step,
    whole rows in VMEM. ``inputs`` (each ``[dtype, kind]``), ``nodes`` (as
    an elementwise kernel's, each with the ``kind`` of its value) and
    ``outputs`` (each ``[value, dtype, kind]``); a kind is ``full`` (a block
    of rows), ``row`` (a value a row), ``lanes`` (a value a column, the same
    for every row) or ``scalar``. Its scalars are the operands' rows, then
    the outputs'."""
    rows, n, bm = spec["height"], spec["n"], spec["block"]
    shapes = {"full": (bm, n), "row": (bm, 1), "lanes": (1, n), "scalar": (1, 1)}
    views = {"full": (rows, n), "row": (rows, 1), "lanes": (n,)}
    blocks = {
        "full": pl.BlockSpec((bm, n), lambda i: (i, 0)),
        "row": pl.BlockSpec((bm, 1), lambda i: (i, 0)),
        "lanes": pl.BlockSpec((n,), lambda i: (0,)),
    }
    inputs, outputs = spec["inputs"], spec["outputs"]

    def kernel(refs, memory, result):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        piped, specs, scalars = [], [], {}
        for i, (dtype, kind) in enumerate(inputs):
            if kind == "scalar":
                scalars[i] = pl.scratch(pl.VMEM((1,), dtype))
                copy(pallas.view(memory, refs[i], (), dtype), scalars[i])
                continue
            piped.append(pallas.view(memory, refs[i], views[kind], dtype))
            specs.append(blocks[kind])
        read = {i: prims.broadcast_in_dim(s[...], (1, 1), (1,)) for i, s in scalars.items()}
        outs = [
            pallas.view(result, refs[len(inputs) + k], views[kind], dtype)
            for k, (_, dtype, kind) in enumerate(outputs)
        ]

        def body(*refs_):
            given = iter(refs_[: len(piped)])
            values = []
            for i, (_, kind) in enumerate(inputs):
                if i in read:
                    values.append(read[i])
                    continue
                v = next(given)[...]
                values.append(prims.reshape(v, (1, n)) if kind == "lanes" else v)
            values = replayed(spec["nodes"], values, shapes)
            for ref, (k, _, _) in zip(refs_[len(piped) :], outputs):
                ref[...] = values[k]

        o_specs = [blocks[kind] for _, _, kind in outputs]
        pl.emit_pipeline(body, grid=(rows // bm,), in_specs=specs, out_specs=o_specs)(*piped, *outs)

    return pallas.program(kernel, spec)
