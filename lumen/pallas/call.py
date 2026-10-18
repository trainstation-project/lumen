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

from .pipeline import _NAMES, BoundedSlice

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


def pallas_call(
    kernel,
    out_shape,
    *,
    grid=(),
    in_specs=None,
    out_specs=None,
    scratch_shapes=(),
    input_output_aliases=None,
    name=None,
):
    """``kernel`` as an op: a function of the operands, returning outputs of
    ``out_shape`` (a ``ShapeDtypeStruct``, or a list or tuple of them)
    (``pl.pallas_call``). An output of ``input_output_aliases``
    (``{input: output}``) starts as its input (which JAX donates)."""
    aliases = dict(input_output_aliases or {})
    if in_specs is not None and not isinstance(in_specs, (list, tuple)):
        raise TypeError("`in_specs` must be a tuple or a list")
    for spec in _specs(in_specs, 0) + _specs(out_specs, 0):
        for d in getattr(spec, "block_shape", None) or ():
            if isinstance(d, BoundedSlice):
                raise NotImplementedError(f"Unsupported block dimension type: {type(d).__name__}")
    single = not isinstance(out_shape, (list, tuple))
    outs = [out_shape] if single else list(out_shape)
    grid = (grid,) if isinstance(grid, int) else tuple(grid)
    # A named grid's axes: (name, size) each.
    names = [a[0] for a in grid] if grid and all(isinstance(a, tuple) for a in grid) else []
    grid = tuple(a[1] for a in grid) if names else grid
    label = f"pallas_call[{name or getattr(kernel, '__name__', 'kernel')}]"

    def call(*args):
        if isinstance(out_specs, (list, tuple)) and not single and len(out_specs) != len(outs):
            raise ValueError(
                "Pytree for `out_specs` and `out_shape` do not match. There are 1 mismatches, including:\n"
                f"    * `out_specs` is a tuple of length {len(out_specs)} but `out_shape` is a tuple of length {len(outs)}"
            )
        if not outs:
            # Nothing it writes is read: no op (JAX's returns no outputs).
            return ()
        args = [tracer._lift(a) for a in args]
        types = [(a.dtype, tuple(a.shape)) for a in args] + [(o.dtype, o.shape) for o in outs]
        _check_aliases(aliases, types[: len(args)], types[len(args) :])

        def build(spec):
            from . import kernel as trace

            def body(scalars, memory, result):
                from lumen.ops.tpu.pallas import view

                refs = [
                    view(memory if i < len(args) else result, scalars[i], shape, dtype)
                    for i, (dtype, shape) in enumerate(types)
                ]
                if aliases:
                    from . import SemaphoreType, make_async_copy, scratch

                    sem = scratch(SemaphoreType.DMA)
                    for i, o in aliases.items():
                        c = make_async_copy(refs[i], refs[len(args) + o], sem)
                        c.start()
                        c.wait()
                _NAMES.append(names)
                try:
                    _run(kernel, refs[: len(args)], refs[len(args) :], grid, in_specs, out_specs, scratch_shapes)
                finally:
                    _NAMES.pop()

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


def where(f):
    """Function ``f``'s name and where it is defined (for JAX's messages)."""
    code = getattr(f, "__code__", None)
    name = getattr(f, "__name__", repr(f))
    return f"{name} at {code.co_filename}:{code.co_firstlineno}" if code else name


def _call(kernel, *refs):
    """``kernel(*refs)``, which returns nothing."""
    if kernel(*refs) is not None:
        raise ValueError(f"The kernel function of pallas_call {where(kernel)} should return None")


def _check_aliases(aliases, ins, outs):
    """Errors (JAX's) for ``aliases`` of inputs ``ins`` and outputs ``outs``
    (dtype and shape each) out of range or of different types."""

    def abstract(t):
        return f"{t[0]}[{','.join(map(str, t[1]))}]"

    for i, o in aliases.items():
        mapping = f"input_output_aliases contains the mapping '{i}:{o}'"
        if not 0 <= i < len(ins):
            raise ValueError(f"{mapping} with input index {i} outside the range [0, {len(ins)})")
        if not 0 <= o < len(outs):
            raise ValueError(f"{mapping} with output index {o} outside the range [0, {len(outs)})")
        if (ins[i][0], tuple(ins[i][1])) != (outs[o][0], tuple(outs[o][1])):
            raise ValueError(
                f"{mapping} referring to input[{i}] with abstract value {abstract(ins[i])} and to "
                f"output[{o}] with a different abstract value {abstract(outs[o])}"
            )


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
        # Without a grid, a block of no shape is the operand whole: staged
        # as it is (one step).
        if (
            space is SMEM
            or (space is VMEM or whole or spec is None or not grid)
            and (spec is None or spec.block_shape is None)
        ):
            staged.append((i, SMEM if space is SMEM else VMEM))
        else:
            piped.append(i)
    sem = scratch(SemaphoreType.DMA) if staged else None

    def copy(src, dst):
        c = make_async_copy(src, dst, sem)
        c.start()
        c.wait()

    for i, space in staged:
        args[i] = scratch(space(refs[i].kernel.ref_type(refs[i].var)[1], refs[i].dtype))
        if refs[i]._scalar:
            args[i] = type(args[i])(args[i].kernel, args[i].var, shape=())
        if i < len(ins):
            copy(refs[i], args[i])
    extra = [scratch(s) for s in scratch_shapes]
    if piped or grid:
        specs = [s if s is not None else BlockSpec() for s in specs]

        def body(*blocks):
            for i, b in zip(piped, blocks):
                args[i] = b
            _call(kernel, *args, *extra)

        n_in = sum(1 for i in piped if i < len(ins))
        labels = [f"args[{i}]" if i < len(ins) else f"outputs[{i - len(ins)}]" for i in piped]
        emit_pipeline(
            body,
            grid=grid,
            in_specs=[specs[i] for i in piped[:n_in]],
            out_specs=[specs[i] for i in piped[n_in:]],
            labels=labels,
        )(*[refs[i] for i in piped])
    else:
        _call(kernel, *args, *extra)
    for i, _ in staged:
        if i >= len(ins):
            copy(args[i], refs[i])
