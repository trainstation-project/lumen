"""``pallas_call``: a kernel of lumen's Pallas as an op of a traced program,
on tensors, as JAX's ``pl.pallas_call`` is an op of a jitted function.

Inside ``lumen.compile``, ``pallas_call(kernel, out_shape, ...)(*xs)``
records a ``pallas_call`` node (a graph primitive, run on a TPU): its kernel
is traced later, when the plan first runs it, on the memory the plan placed
its operands and outputs in. Each operand and output is then an HBM ref of
its shape and dtype, a view of its rows of the device's memory; the kernel
sees, as in JAX:

* without ``in_specs``/``out_specs`` and a grid, each copied to VMEM whole
  (the outputs copied back after);
* with them, the blocks of a pipeline over ``grid``
  (``emit_pipeline``), an operand of ``BlockSpec(memory_space=ANY)`` its HBM
  ref as it is;

then a ref of each of ``scratch_shapes``.

Operands and outputs are viewed in place, in a TPU's tiles
(``device::tpu::layout``, as lumen keeps a TPU's tensors): any shape, any
dtype a TPU stores.
"""

from lumen.graph import prims, tracer

# The kernels of the pallas_calls traced so far, by handle: each a function
# of a spec (the program's scalars and rows) to its Mosaic.
_CALLS = []

ANY = "any"


class ShapeDtypeStruct:
    """An output's ``shape`` and ``dtype`` (``jax.ShapeDtypeStruct``)."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = tuple(shape), str(dtype)


def program(spec):
    """The Mosaic of pallas_call kernel ``spec["kernel"]`` for ``spec``
    (Rust's entry: ``lumen/ops/tpu``)."""
    return _CALLS[spec["kernel"]](spec)


def pallas_call(kernel, out_shape, *, grid=(), in_specs=None, out_specs=None, scratch_shapes=(), name=None):
    """``kernel`` as an op: a function of the operands, returning outputs of
    ``out_shape`` (a ``ShapeDtypeStruct``, or a list or tuple of them)
    (``pl.pallas_call``)."""
    single = not isinstance(out_shape, (list, tuple))
    outs = [out_shape] if single else list(out_shape)
    grid = (grid,) if isinstance(grid, int) else tuple(grid)
    label = f"pallas_call[{name or getattr(kernel, '__name__', 'kernel')}]"

    def call(*args):
        args = [tracer._lift(a) for a in args]
        types = [(a.dtype, tuple(a.shape)) for a in args] + [(o.dtype, o.shape) for o in outs]

        def build(spec):
            from . import kernel as trace

            def body(scalars, memory, result):
                from lumen.ops.tpu.pallas import view

                refs = [
                    view(memory if i < len(args) else result, scalars[i], shape, dtype)
                    for i, (dtype, shape) in enumerate(types)
                ]
                _run(kernel, refs[: len(args)], refs[len(args) :], grid, in_specs, out_specs, scratch_shapes)

            return trace(body, scalars=spec["scalars"], rows=spec["rows"], host_rows=spec["host_rows"], name=label)

        handle = len(_CALLS)
        _CALLS.append(build)
        first = prims.bind(
            "pallas_call", *args, op=label, kernel=handle, outputs=[(o.dtype, list(o.shape)) for o in outs]
        )
        rest = [
            prims.bind("fusion_output", first, index=k, dtype=o.dtype, shape=o.shape)
            for k, o in enumerate(outs[1:], 1)
        ]
        # A tuple for a list too (as JAX: "we normalize out_shape to a tuple").
        return first if single else (first, *rest)

    return call


def _specs(specs, count):
    if specs is None:
        return [None] * count
    return list(specs) if isinstance(specs, (list, tuple)) else [specs]


def _run(kernel, ins, outs, grid, in_specs, out_specs, scratch_shapes):
    """``kernel`` on HBM refs ``ins`` and ``outs``, as ``pallas_call`` runs it:
    each operand staged whole in VMEM (no specs and no grid; a spec of
    ``memory_space=VMEM`` and no block shape) or SMEM (``memory_space=SMEM``),
    its HBM ref as it is (``memory_space=ANY``), or its blocks pipelined over
    the grid (a spec with a block shape)."""
    from . import SMEM, VMEM, SemaphoreType, emit_pipeline, make_async_copy, scratch
    from .pipeline import BlockSpec

    refs = ins + outs
    whole = not grid and in_specs is None and out_specs is None
    specs = _specs(in_specs, len(ins)) + _specs(out_specs, len(outs))
    args, piped, staged = list(refs), [], []
    for i, spec in enumerate(specs):
        space = None if spec is None else spec.memory_space
        if space == ANY:
            continue
        if space is SMEM or (space is VMEM or whole or spec is None) and (spec is None or spec.block_shape is None):
            staged.append((i, SMEM if space is SMEM else VMEM))
        else:
            piped.append(i)
    sem = scratch(SemaphoreType.DMA) if staged else None

    def copy(src, dst):
        c = make_async_copy(src, dst, sem)
        c.start()
        c.wait()

    for i, space in staged:
        args[i] = scratch(space(refs[i].shape, refs[i].dtype))
        if i < len(ins):
            copy(refs[i], args[i])
    extra = [scratch(s) for s in scratch_shapes]
    if piped or grid:
        specs = [s if s is not None else BlockSpec() for s in specs]

        def body(*blocks):
            for i, b in zip(piped, blocks):
                args[i] = b
            kernel(*args, *extra)

        n_in = sum(1 for i in piped if i < len(ins))
        emit_pipeline(
            body,
            grid=grid,
            in_specs=[specs[i] for i in piped[:n_in]],
            out_specs=[specs[i] for i in piped[n_in:]],
        )(*[refs[i] for i in piped])
    else:
        kernel(*args, *extra)
    for i, _ in staged:
        if i >= len(ins):
            copy(args[i], refs[i])
