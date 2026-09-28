from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
from ...ops import register


OP = "lumen::fill_"

WARP_SIZE = 32
LOG_WARP_SIZE = 5
DEFAULT_BLOCK_SIZE = 256
DEFAULT_M = 1
BLOCK_SIZES = (128, 256, 512, 1024)
VECTOR_BYTES = 16


class CuteFill:
    def __init__(self, BLOCK_SIZE: int = DEFAULT_BLOCK_SIZE, M: int = DEFAULT_M) -> None:
        self.BLOCK_SIZE = BLOCK_SIZE
        self.M = M

    @cute.kernel
    def kernel(
        self,
        gT: cute.Tensor,
        gC: cute.Tensor,
        value: Any,
        TOTAL_TILES: Int32,
        numel: Int32,
    ) -> None:
        BLOCK_ID, _, _ = cute.arch.block_idx()
        NUM_BLOCKS, _, _ = cute.arch.grid_dim()
        THREAD_ID, _, _ = cute.arch.thread_idx()

        while BLOCK_ID < TOTAL_TILES:
            block_coord = ((None, None, None), BLOCK_ID)

            blkT = gT[block_coord]
            blkC = gC[block_coord]

            thrT = blkT[THREAD_ID, None, None]
            thrC = blkC[THREAD_ID, None, None]

            size = cute.size(thrT)

            if cute.elem_less(thrC[0], numel):
                if cute.elem_less(thrC[size - 1], numel):
                    # The common case: the whole vector is in bounds.
                    for i in range(cute.size(thrT)):
                        thrT[i] = value
                else:
                    for i in range(cute.size(thrT)):
                        if cute.elem_less(thrC[i], numel):
                            thrT[i] = value

            BLOCK_ID += NUM_BLOCKS

    @cute.jit
    def __call__(
        self,
        mT: cute.Tensor,
        stream: cuda.CUstream,
        numel: Int32
    ) -> None:
        vector_size = self._vector_size(mT)

        thr_layout = cute.make_ordered_layout(
            (self.BLOCK_SIZE >> LOG_WARP_SIZE, WARP_SIZE), order=(1, 0)
        )
        val_layout = cute.make_ordered_layout((self.M, vector_size), order=(1, 0))
        tiler, _ = cute.make_layout_tv(thr_layout, val_layout)

        gT = cute.zipped_divide(mT, tiler)
        gC = cute.zipped_divide(cute.make_identity_tensor(mT.shape), tiler)

        TOTAL_TILES = cute.size(gC, mode=[1])
        NUM_BLOCKS = cute.arch.get_device_properties().multi_processor_count

        self.kernel(
            gT,
            gC,
            self._fill_value,
            TOTAL_TILES,
            numel
        ).launch(
            grid=(NUM_BLOCKS, 1, 1),
            block=(self.BLOCK_SIZE, 1, 1),
            stream=stream
        )

    def _vector_size(self, mT: cute.Tensor) -> int:
        return max(1, VECTOR_BYTES // (mT.element_type.width // 8))

    @property
    def _value(self) -> Any:
        return self._fill_value

    def launch(self, cute_tensor: cute.Tensor, cute_value: Any, stream: cuda.CUstream, numel: int) -> None:
        self._fill_value = cute_value

        key = (self.BLOCK_SIZE, self.M, numel)
        cache = type(self).__dict__.setdefault("_compiled", {})
        compiled = cache.get(key)
        if compiled is None:
            compiled = cute.compile(self, cute_tensor, stream, numel)
            cache[key] = compiled

        compiled(cute_tensor, stream, numel)


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

_KERNELS: dict[tuple[int, int], CuteFill] = {}


def _cutlass_value(dtype: str, value: Any):
    """``value`` converted to ``dtype``, ready to hand to the kernel.

    Converting here means the DSL sees the tensor's own element type, so
    a narrow dtype is not silently widened (or rejected) by the DSL
    picking its own default. ``bfloat16`` has no Python code to pack it
    with, and is refused rather than quietly filled with something else.
    """
    from cutlass import BFloat16, Float16, Float32

    if dtype not in _STRUCT_CODES:
        raise TypeError(
            f"cannot fill a {dtype} tensor with the CuTe DSL kernel: its "
            "element type has no host-side conversion. Nothing is silently "
            "narrowed -- the built-in CUDA kernel is the fallback."
        )

    if dtype == "bool":
        return bool(value)
    if dtype == "float32":
        return Float32(value)
    if dtype == "float16":
        return Float16(value)
    if dtype.startswith("float"):
        return float(value)
    if dtype.startswith("bfloat"):
        return BFloat16(value)
    return int(value)


def _numel(t) -> int:
    """``t``'s element count. ``numel`` is a property on ``lumen.Tensor``."""
    n = t.numel
    return int(n() if callable(n) else n)


#: What this kernel requires of a tensor, as ``(requirement, predicate,
#: why)``. A CuTe DSL kernel is compiled around one static layout and one
#: store width, so anything that changes the layout at run time has to be
#: refused rather than mis-filled. Each entry pairs the check with the
#: reason it exists, so the error can say what is wrong *and* which way
#: out is available.
_PRECONDITIONS = (
    (
        "its device is CUDA",
        lambda t, _: str(t.device).startswith("cuda"),
        "the kernel is compiled for the CUDA target, so it cannot run on "
        "{device}. Register it for that device instead, or let the "
        "built-in kernel handle it.",
        TypeError,
    ),
    (
        "it is contiguous",
        lambda t, _: t.is_contiguous(),
        "the kernel is compiled around a static layout and a {bytes}-byte "
        "store, so a strided view would be filled incorrectly. Call "
        ".contiguous() first, or unregister the Python kernel so the "
        "built-in CUDA fill_ handles the view.",
        ValueError,
    ),
)


def _check(t, value: Any = None) -> None:
    """Raise unless ``t`` satisfies every entry in :data:`_PRECONDITIONS`.

    A ``TypeError`` for a kind of tensor the kernel cannot take at all, and
    a ``ValueError`` for a tensor it could take if it were laid out
    differently -- so a caller can tell "wrong call" from "wrong data".
    """
    for requirement, holds, why, error in _PRECONDITIONS:
        if holds(t, value):
            continue
        reason = why.format(device=t.device, bytes=VECTOR_BYTES)
        raise error(
            f"the CuTe DSL fill kernel needs a tensor, but {requirement}: {reason}"
        )


def fill(t, value: Any) -> None:
    _check(t, value)

    numel = _numel(t)

    dtype = str(t.dtype)
    key = (DEFAULT_BLOCK_SIZE, DEFAULT_M)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _KERNELS[key] = CuteFill(*key)

    stream = cuda.CUstream(0)

    # `t` exports a DLPack capsule; `from_dlpack` wraps it without copying,
    # so the CuTe tensor aliases `t`'s storage. The alignment hint is what
    # lets the compiler emit the wide stores this kernel is shaped around;
    # a contiguous tensor views its allocation from offset 0, and lumen's
    # CUDA allocations are 256-byte aligned (lumen/allocator/cuda.rs), so
    # the promised 16 bytes hold.
    cute_tensor = from_dlpack(t, assumed_align=VECTOR_BYTES)

    kernel.launch(cute_tensor, _cutlass_value(dtype, value), stream, numel)


register(OP, "cuda", fill)
