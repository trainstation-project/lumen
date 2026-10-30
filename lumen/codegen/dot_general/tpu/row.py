"""Pallas program for a TPU dot whose epilogue reduces along rows, so a block spans all ``n`` columns."""

from lumen import pallas as pl
from lumen.codegen.utils.tpu import per_row
from lumen.kernels.tpu import pallas


def program(spec):
    """The dot of ``spec`` over ``[bm, n]`` row blocks: ``height`` rows, ``n``, ``k``, ``transposed``,
    ``blocks`` (bm, bk), ``inputs`` and ``outputs`` (each full, per row, per lane or scalar) and the
    per-row epilogue ``nodes``."""
    rows, n, k, t = spec["height"], spec["n"], spec["k"], spec["transposed"]
    bm, bk = spec["blocks"]
    dtype, value = spec["operands"], spec["value"]
    inputs, outputs = spec["inputs"], spec["outputs"]
    kb = k // bk
    shapes = {"full": (bm, n), "row": (bm, 1), "lanes": (1, n), "scalar": (1, 1)}
    views = {"full": (rows, n), "row": (rows, 1), "lanes": (n,)}
    blocks = {
        "full": pl.BlockSpec((bm, n), lambda i, s: (i, 0)),
        "row": pl.BlockSpec((bm, 1), lambda i, s: (i, 0)),
        "lanes": pl.BlockSpec((n,), lambda i, s: (0,)),
    }

    def kernel(refs, memory, result, *scratch):
        """Pipelines the row-blocked dot and its epilogue from ``memory`` into ``result``."""

        def copy(src, dst):
            """Blocking DMA of ``src`` into ``dst``."""
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        x = pallas.view(memory, refs[0], (rows, k), dtype)
        y = pallas.view(memory, refs[1], (n, k) if t else (k, n), dtype)
        piped, specs, scalars = [], [], {}
        for i, (d, kind) in enumerate(inputs):
            if kind == "scalar":
                scalars[i] = pl.scratch(pl.VMEM((1,), d))
                copy(pallas.view(memory, refs[2 + i], (), d), scalars[i])
                continue
            piped.append(pallas.view(memory, refs[2 + i], views[kind], d))
            specs.append(blocks[kind])
        read = {i: s[...].expand((1, 1)) for i, s in scalars.items()}
        outs = [
            pallas.view(result, refs[2 + len(inputs) + o], views[kind], d) for o, (_, d, kind) in enumerate(outputs)
        ]

        def finish(product, given, written):
            """Runs the epilogue on ``product`` with the inputs' blocks and writes the outputs."""
            given = iter(given)
            values = [product.to(value)]
            for i, (_, kind) in enumerate(inputs):
                if i in read:
                    values.append(read[i])
                    continue
                v = next(given)[...]
                values.append(v.reshape(1, n) if kind == "lanes" else v)
            values = per_row(spec["nodes"], values, shapes)
            for ref, (v, _, _) in zip(written, outputs):
                ref[...] = values[v]

        def body(xb, yb, *refs_):
            """One grid step: accumulate this K block's product, finishing on the last."""
            given, written = refs_[: len(piped)], refs_[len(piped) :]
            product = pl.dot(xb[...], yb[...], trans_b=t)
            if kb == 1:
                finish(product, given, written)
                return
            step = pl.program_id(1)

            @pl.when(step == 0)
            def _():
                """Zeroes the accumulator on the first K step."""
                scratch[0][...] = 0.0

            scratch[0][...] += product

            @pl.when(step == kb - 1)
            def _():
                """Runs the epilogue after the last K step."""
                finish(scratch[0][...], given, written)

        x_spec = pl.BlockSpec((bm, bk), lambda i, s: (i, s))
        y_spec = pl.BlockSpec((n, bk), lambda i, s: (0, s)) if t else pl.BlockSpec((bk, n), lambda i, s: (s, 0))
        o_specs = [blocks[kind] for _, _, kind in outputs]
        pl.emit_pipeline(body, grid=(rows // bm, kb), in_specs=[x_spec, y_spec, *specs], out_specs=o_specs)(
            x, y, *piped, *outs
        )

    scratch = [pl.VMEM((bm, n), "float32")] if kb > 1 else []
    return pallas.program(kernel, spec, scratch=scratch)
