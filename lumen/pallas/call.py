"""``pallas_call``: a Pallas kernel as a traced op, lowered to Mosaic when the TPU plan builds it."""

from lumen.graph import prims, tracer

from .pipeline import _NAMES, BoundedSlice

# Builders of each pallas_call's Mosaic program, indexed by the handle stored in its node.
_CALLS = []

# ``memory_space`` of an operand left in HBM, passed to the kernel as is.
ANY = "any"


class ShapeDtypeStruct:
    """An output's ``shape`` and ``dtype`` (``jax.ShapeDtypeStruct``)."""

    def __init__(self, shape, dtype):
        """Records the output's shape and dtype."""
        self.shape, self.dtype = tuple(shape), str(dtype)


def program(spec):
    """The Mosaic of pallas_call kernel ``spec["kernel"]`` for ``spec``
    (Rust's entry: ``lumen/kernels/tpu``)."""
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
    # Named grid axes, ``(name, size)`` pairs, for ``pl.program_id`` by name.
    names = [a[0] for a in grid] if grid and all(isinstance(a, tuple) for a in grid) else []
    grid = tuple(a[1] for a in grid) if names else grid
    label = f"pallas_call[{name or getattr(kernel, '__name__', 'kernel')}]"

    def call(*args):
        """Records the kernel as one ``pallas_call`` node on ``args`` and returns its outputs."""
        if isinstance(out_specs, (list, tuple)) and not single and len(out_specs) != len(outs):
            raise ValueError(
                "Pytree for `out_specs` and `out_shape` do not match. There are 1 mismatches, including:\n"
                f"    * `out_specs` is a tuple of length {len(out_specs)} but `out_shape` is a tuple of length {len(outs)}"
            )
        if not outs:
            return ()
        args = [tracer._lift(a) for a in args]
        types = [(a.dtype, tuple(a.shape)) for a in args] + [(o.dtype, o.shape) for o in outs]
        _check_aliases(aliases, types[: len(args)], types[len(args) :])

        def build(spec):
            """Traces the kernel into a Mosaic program for the launch ``spec``."""
            from . import kernel as trace

            def body(scalars, memory, result):
                """Views the scalar-addressed rows as refs, applies aliases, then runs the kernel."""
                from lumen.kernels.tpu.pallas import view

                refs = [
                    view(memory if i < len(args) else result, scalars[i], shape, dtype)
                    for i, (dtype, shape) in enumerate(types)
                ]
                # An aliased output starts as a copy of its input.
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
        # Outputs after the first are projections of the one node.
        rest = [
            prims.bind("fusion_output", first, index=k, dtype=o.dtype, shape=o.shape)
            for k, o in enumerate(outs[1:], 1)
        ]
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
        """``dtype[d0,d1,...]``, as JAX prints an abstract value."""
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
    """``specs`` as a list (``count`` Nones for None, a single spec wrapped)."""
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
        # Operands without a block shape (or without a grid) are staged whole; others are pipelined.
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
        """Blocking DMA of ``src`` into ``dst``."""
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
            """One pipeline step: substitutes the current blocks and runs the kernel."""
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
    # Staged outputs are copied back to HBM after the kernel.
    for i, _ in staged:
        if i >= len(ins):
            copy(args[i], refs[i])
