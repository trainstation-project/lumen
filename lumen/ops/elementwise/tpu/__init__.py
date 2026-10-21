import math

from lumen import pallas as pl
from lumen.codegen.tpu import computed, indexed
from lumen.ops.tpu import pallas


def _indexed(spec):
    """A loop with layout primitives: the output in blocks of ``blocks``
    over ``grid``; each of ``inputs`` (``[dtype, shape, block, index]``,
    ``index`` a ``[grid axis, add]`` a dimension, ``-1`` a fixed block) read
    in blocks of its own, so broadcasts, slices and transposes of major
    dimensions are where it reads."""
    shape, grid, blocks, inputs = spec["shape"], tuple(spec["grid"]), spec["blocks"], spec["inputs"]

    def block_spec(block, index):
        return pl.BlockSpec(
            tuple(block), lambda *ids, index=index: tuple(ids[d] + a if d >= 0 else a for d, a in index)
        )

    out_view, out_index = tuple(shape), [[d, 0] for d in range(len(shape))]
    if spec["window"]:
        whole, start = spec["window"]
        out_view = (*shape[:-1], whole)
        out_index[-1] = [len(shape) - 1, start // blocks[-1]]

    def kernel(rows, memory, result):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        piped, specs, read = [], [], {}
        for i, (dtype, view, block, index) in enumerate(inputs):
            if not view:
                v = pl.scratch(pl.VMEM((1,), dtype))
                copy(pallas.view(memory, rows[i], (), dtype), v)
                read[i] = v[...]
                continue
            piped.append(pallas.view(memory, rows[i], tuple(view), dtype))
            specs.append(block_spec(block, index))
        out = pallas.view(result, rows[len(inputs)], out_view, spec["output"])

        def body(*refs):
            ids = [pl.program_id(a) for a in range(len(grid))]
            given = iter(refs[:-1])
            values = [read[i] if i in read else next(given)[...] for i in range(len(inputs))]
            refs[-1][...] = indexed(spec["nodes"], values, ids)[spec["result"]]

        out_spec = block_spec(blocks, out_index)
        pl.emit_pipeline(body, grid=grid, in_specs=specs, out_specs=[out_spec])(*piped, out)

    return pallas.program(kernel, spec)


def program(spec):
    """The kernel of ``spec``: ``shape``, ``inputs`` (the operands' dtypes),
    ``broadcast`` (whether each operand is 0-d, the output not), ``nodes``
    (each a primitive's ``name``, ``params`` and ``args``: indices of the
    values, the operands then the nodes before), ``results`` (the indices
    of the values written, an output each: a horizontal fusion's several),
    ``outputs`` (their dtypes),
    ``blocks`` (of the leading dimensions together and the last two, or of
    the elements) and
    ``window`` (``[n, start]``: the output written at columns ``start`` on
    of a buffer ``n`` wide; none: a buffer of its own).
    Its scalars are the operands' rows, then the output's. With
    ``indexed``, a loop with layout primitives (``_indexed``)."""
    if spec.get("indexed"):
        return _indexed(spec)
    shape, dtypes, broadcast = spec["shape"], spec["inputs"], spec["broadcast"]
    if len(shape) >= 2:
        (bl, bm, bn), (m, n) = spec["blocks"], shape[-2:]
        view = (math.prod(shape[:-2]), m, n)
        block_shape = (bl, bm, bn) if bl > 1 else (bm, bn)
        block = pl.BlockSpec((bl if bl > 1 else None, bm, bn), lambda l, i, j: (l, i, j))
        grid = (view[0] // bl, m // bm, n // bn)
    else:
        (b,) = spec["blocks"]
        view, block_shape = (max(math.prod(shape), 1),), (b,)
        block = pl.BlockSpec((b,), lambda i: (i,))
        grid = (view[0] // b,)
    out_view, out_block = view, block
    if spec["window"]:
        whole, start = spec["window"]
        out_view = (view[0], m, whole)
        out_block = pl.BlockSpec((bl if bl > 1 else None, bm, bn), lambda l, i, j: (l, i, j + start // bn))

    scalar = not shape

    shape_of = (1,) if scalar else block_shape

    def apply(values):
        values = computed(spec["nodes"], values, shape_of)
        return [values[r] for r in spec["results"]]

    def kernel(rows, memory, result):
        def copy(src, dst):
            c = pl.make_async_copy(src, dst, pl.scratch(pl.SemaphoreType.DMA))
            c.start()
            c.wait()

        piped, scalars = [], {}
        for i, dtype in enumerate(dtypes):
            if not (broadcast[i] or scalar):
                piped.append(pallas.view(memory, rows[i], view, dtype))
                continue
            scalars[i] = pl.scratch(pl.VMEM((1,), dtype))
            copy(pallas.view(memory, rows[i], (), dtype), scalars[i])
        read = {i: s[...] for i, s in scalars.items()}
        outputs = spec["outputs"]
        outs = [
            pallas.view(result, rows[len(dtypes) + k], () if scalar else (out_view if k == 0 else view), dtype)
            for k, dtype in enumerate(outputs)
        ]
        if scalar:
            vmem = pl.scratch(pl.VMEM((1,), outputs[0]))
            vmem[...] = apply(read[i] for i in range(len(dtypes)))[0]
            copy(vmem, outs[0])
            return

        def body(*blocks):
            given = iter(blocks)
            values = [read[i] if i in read else next(given)[...] for i in range(len(dtypes))]
            for ref, value in zip(blocks[len(piped) :], apply(values)):
                ref[...] = value

        o_specs = [out_block] + [block] * (len(outputs) - 1)
        pl.emit_pipeline(body, grid=grid, in_specs=[block] * len(piped), out_specs=o_specs)(*piped, *outs)

    return pallas.program(kernel, spec)
