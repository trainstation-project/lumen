"""lumen's TPU kernels on a device's memory, lowered for Rust.

A TPU kernel is a module of ``lumen.pallas`` source beside its op
(``lumen/ops/<op>/tpu.py``) defining ``program(spec)``: its kernel for
``spec`` (a dict the Rust side chose: shapes, blocks, packing; and the
program's ``scalars`` count, ``host_rows`` and the memory's ``rows``),
traced and lowered to Mosaic by :func:`program` here. Rust
(``lumen/ops/tpu/mod.rs``) calls :func:`lower` once per kernel and spec,
wraps the Mosaic in a program, compiles it with libtpu and runs it.

Every kernel works on a device's whole memory: the TPU allocator's one
buffer, ``uint32[rows, 1, 128]`` (rows of 512 bytes, laid out as the host
lays out bytes), each tensor in it in the TPU's tiles (:func:`view`). Its
arguments are int32 scalars (the rows of the tensors it reads and writes,
first), optionally host bytes copied to the TPU
(``uint32[host_rows, 1, 128]``), and the memory, as read and as written.
"""

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
    # Where it is, to view it as another shape of the same bytes.
    ref._view = (memory, row)
    return ref
