from __future__ import annotations

from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from ..ops import register


OP = "lumen::fill_"
_THREADS = 256
_VECTOR_BYTES = 16


@cute.kernel
def _fill_kernel(
    gT: cute.Tensor,
    value: cutlass.Numeric,
    numel: cutlass.Int32,
) -> None:
    """Set every element of ``gT`` to ``value``.

    ``gT`` is the flat tensor partitioned as ``((V, T), tile)``: mode ``0``
    is one block's tile of ``(V, T)`` -- ``V`` contiguous elements per
    thread, ``T`` threads -- and mode ``1`` indexes the tile, i.e. the
    block. So block ``bidx`` owns elements
    ``[bidx * V * T, (bidx + 1) * V * T)``, and thread ``tidx`` writes
    ``[bidx * V * T + tidx * V, ... + V)``.

    ``numel`` is passed explicitly because ``gT``'s cosize is rounded up
    to a whole number of tiles: the trailing tile is partial whenever the
    element count is not a multiple of ``V * T``, and the guard below is
    what makes any shape of tensor fillable.

    ``value`` already has the tensor's element type -- :func:`_packed_value`
    converts it on the host -- so this body carries no dtype dispatch.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # This block's tile, then this thread's vector within it. After the
    # second slice the remaining modes are `(V,)`, so `thr[i]` is the
    # i-th element this thread owns.
    blk = gT[None, bidx]
    thr = blk[tidx, None]

    vector = cute.size(thr)
    base = (bidx * cute.size(gT, mode=[0]) + tidx * vector)

    # A fill needs no staging through shared memory: the value is already
    # in a register, so each thread issues one wide store. The guard is
    # per element rather than per vector because the final tile is
    # partial for any element count that is not a whole number of tiles.
    for i in range(vector):
        if base + i < numel:
            thr[i] = value


@cute.jit
def _fill_launch(
    gT: cute.Tensor,
    value: cutlass.Numeric,
    numel: cutlass.Int32,
    tiles: cutlass.Int32,
) -> None:
    """Launch :func:`_fill_kernel` over ``gT``, one block per tile.

    ``gT`` arrives already partitioned as ``((V, T), tile)`` (the host
    does that with ``logical_divide``, so the layout is static), which
    makes the block shape readable off the tensor itself.
    """
    _fill_kernel(gT, value, numel).launch(
        grid=(tiles, 1, 1),
        block=(cute.size(gT, mode=[0, 1]), 1, 1),
    )


# ---------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------

#: ``struct`` code points, by lumen dtype. Used by :func:`_cutlass_value`
#: to build a scalar the DSL accepts without narrowing it to something the
#: kernel was not compiled for.
_STRUCT_CODES = {
    "bool": "?",
    "uint8": "B",
    "int8": "b",
    "uint16": "H",
    "int16": "h",
    "uint32": "I",
    "int32": "i",
    "uint64": "Q",
    "int64": "q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}

#: Bytes per element, by lumen dtype.
_ELEMENT_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
}


def vector_length(dtype: str) -> int:
    """Elements per thread for ``dtype``: as many as fit in a 128-bit store.

    ``float32`` gives 4, ``float16`` gives 8, and so on. This is the
    element count that makes each thread issue one wide store, which is
    what a bandwidth-bound fill wants.
    """
    return max(1, _VECTOR_BYTES // _ELEMENT_BYTES[dtype])


def grid_for(numel: int, dtype: str, threads: int = _THREADS) -> tuple[int, int]:
    """The launch shape for ``numel`` elements of ``dtype``: ``(tiles, threads)``.

    ``threads`` is clamped to ``numel`` so a tiny tensor is not launched
    with mostly-idle blocks, and ``tiles`` rounds *up*, because the kernel
    guards the partial trailing tile rather than assuming an exact
    multiple.
    """
    vector = vector_length(dtype)
    threads = max(1, min(threads, (numel + vector - 1) // vector))
    tiles = (numel + threads * vector - 1) // (threads * vector)
    return max(1, tiles), threads


def _packed_value(dtype: str, value: Any) -> bytes:
    """``value`` converted to ``dtype`` and packed, or a ``TypeError``.

    Doing the conversion here means the value reaches the DSL already in
    the tensor's element type, so the kernel body needs no dtype dispatch
    and a fill of ``int32`` compiles once rather than once per dtype.
    """
    try:
        code = _STRUCT_CODES[dtype]
    except KeyError:
        raise TypeError(
            f"cannot fill a {dtype} tensor with the CuTe DSL kernel: "
            "its element type has no Python struct code. Nothing is "
            "silently narrowed."
        ) from None
    import struct

    return struct.pack(code, bool(value) if dtype == "bool" else value)


def _cutlass_value(dtype: str, value: Any) -> "cutlass.Numeric":
    """``value`` as a CuTe numeric of ``dtype``.

    The scalar is built with the tensor's own element type so the kernel
    is compiled for the type it will actually be handed -- passing a
    Python float to a ``float32`` tensor, say, would otherwise let the
    DSL pick its own default type and reject or truncate the value.

    Goes through ``array`` rather than ``struct`` because ``array``
    exposes the element type (``array('f', [x]).typecode``) alongside the
    packed bytes, which is what ``cutlass.Numeric`` wants.
    """
    import array

    try:
        code = _STRUCT_CODES[dtype]
    except KeyError:
        raise TypeError(
            f"cannot fill a {dtype} tensor with the CuTe DSL kernel: its "
            "element type has no host-side code to pack it with. Nothing "
            "is silently narrowed -- fill from the built-in CUDA kernel "
            "instead."
        ) from None

    element = bool(value) if dtype == "bool" else value
    packed = array.array(code, [element])
    return cutlass.Numeric(packed.tobytes(), dtype, signed=_is_signed(dtype))


def _is_signed(dtype: str) -> bool:
    """Whether ``dtype`` is a signed integer type."""
    return dtype.startswith("int")


def fill(t, value: Any) -> None:
    if not t.is_cuda:
        raise TypeError(
            f"the CuTe DSL fill kernel only handles CUDA tensors, got {t.device}"
        )

    numel = t.numel()
    if numel == 0:
        return

    dtype = str(t.dtype)
    scalar = _cutlass_value(dtype, value)

    tiles, threads = grid_for(numel, dtype)

    # Shape the flat tensor as ((V, T), tiles) before it reaches the DSL,
    # so the kernel's slicing is static: mode 0 is one block's tile of
    # (vector, thread), mode 1 selects the block.
    inside = cute.make_layout((vector_length(dtype), threads))
    gT = cute.logical_divide(from_dlpack(t), inside)

    _fill_launch(gT, scalar, numel, tiles)


register(OP, "cuda", fill)
