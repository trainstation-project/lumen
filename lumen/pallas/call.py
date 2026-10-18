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

Operands and outputs are viewed in place: 32-bit dtypes, of two dimensions
or more, the last a multiple of 128 (a TPU lays out narrower dtypes and
shorter rows otherwise than the host's bytes).
"""

import math

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


def _view(memory, row, shape, dtype):
    """The ``shape`` of ``dtype`` at ``row`` of ``memory``: a ref in place."""
    from . import ds

    words = math.prod(shape)
    if dtype not in ("float32", "int32", "uint32") or len(shape) < 2 or shape[-1] % 128:
        raise NotImplementedError(
            f"pallas_call reads and writes 32-bit arrays of two dimensions or more, the last a "
            f"multiple of 128, in place: got {dtype}{list(shape)}"
        )
    return memory.at[ds(row, words // 128)].reshape(*shape).bitcast(dtype)


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
                refs = [
                    _view(memory if i < len(args) else result, scalars[i], shape, dtype)
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
        return first if single else type(out_shape)([first, *rest])

    return call


def _specs(specs, count):
    if specs is None:
        return [None] * count
    return list(specs) if isinstance(specs, (list, tuple)) else [specs]


def _run(kernel, ins, outs, grid, in_specs, out_specs, scratch_shapes):
    """``kernel`` on HBM refs ``ins`` and ``outs``, as ``pallas_call`` runs it."""
    from . import VMEM, SemaphoreType, emit_pipeline, make_async_copy, scratch
    from .pipeline import BlockSpec

    if not grid and in_specs is None and out_specs is None:
        # Each whole, in VMEM.
        sem = scratch(SemaphoreType.DMA)
        vins = [scratch(VMEM(r.shape, r.dtype)) for r in ins]
        vouts = [scratch(VMEM(r.shape, r.dtype)) for r in outs]
        for r, v in zip(ins, vins):
            copy = make_async_copy(r, v, sem)
            copy.start()
            copy.wait()
        kernel(*vins, *vouts, *[scratch(s) for s in scratch_shapes])
        for r, v in zip(outs, vouts):
            copy = make_async_copy(v, r, sem)
            copy.start()
            copy.wait()
        return
    specs = _specs(in_specs, len(ins)) + _specs(out_specs, len(outs))
    refs = ins + outs
    specs = [s if s is not None else BlockSpec() for s in specs]
    piped = [i for i, s in enumerate(specs) if s.memory_space != ANY]
    extra = [scratch(s) for s in scratch_shapes]

    def body(*blocks):
        args = list(refs)
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
