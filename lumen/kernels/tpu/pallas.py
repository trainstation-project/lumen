"""Lowers lumen.pallas TPU kernels to Mosaic for the Rust TPU runtime, and views into device memory."""

import importlib
import json

from lumen import pallas as pl


def lower(module: str, spec: str) -> str:
    """The Mosaic module of ``module``'s kernel for ``spec`` (JSON)."""
    return importlib.import_module(module).program(json.loads(spec))


def program(kernel, spec, scratch=()) -> str:
    """``kernel(scalars, [host,] memory, result, *scratch)`` traced on the
    memory of ``spec`` and lowered to Mosaic."""
    return pl.kernel(
        kernel,
        scalars=spec["scalars"],
        rows=spec["rows"],
        host_rows=spec["host_rows"],
        scratch=scratch,
    )


def view(memory, row, shape, dtype):
    """The ``shape`` of ``dtype`` at ``row`` of ``memory``: a ref of it in
    place, in its TPU layout (``device::tpu::layout``); a scalar's as a
    ``(1,)``'s (the same bytes), which a TPU copies (whole rows), read as
    a scalar."""
    k = memory.kernel
    if tuple(shape) == ():
        return pl.Ref(k, k.view(memory.var, row.var, dtype, [1]), shape=())
    ref = pl.Ref(k, k.view(memory.var, row.var, dtype, list(shape)))
    ref._view = (memory, row)
    return ref
