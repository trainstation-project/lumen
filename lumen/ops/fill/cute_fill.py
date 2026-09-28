"""CUDA ``fill_``: a CuTe DSL kernel, registered for ``lumen::fill_`` on
CUDA when lumen is imported (see ``lumen.ops``).

Only the kernel and its compilation live here. Rust calls :func:`compile`
once per dtype and layout (the view's dimensions in stride order) and
caches the launcher it returns, which it then calls with the data pointer
and the value.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from lumen.ops import register

THREADS = 256

_ELEMENT = {
    "bool": cutlass.Boolean,
    "uint8": cutlass.Uint8,
    "int8": cutlass.Int8,
    "uint16": cutlass.Uint16,
    "int16": cutlass.Int16,
    "uint32": cutlass.Uint32,
    "int32": cutlass.Int32,
    "uint64": cutlass.Uint64,
    "int64": cutlass.Int64,
    "float16": cutlass.Float16,
    "bfloat16": cutlass.BFloat16,
    "float32": cutlass.Float32,
    "float64": cutlass.Float64,
}


@cute.kernel
def _kernel(t: cute.Tensor, value):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * THREADS + tidx
    if i < cute.size(t):
        t[i] = value


def compile(dtype, shape, strides):
    """A launcher for fills of one layout, ``launch(address, value,
    stream)``. The layout is baked into the kernel, and each thread fills
    one element; the first dimension has the smallest stride, so
    neighboring threads store to neighboring addresses."""
    element = _ELEMENT[dtype]
    align = max(1, element.width // 8)

    @cute.jit
    def fill(ptr: cute.Pointer, value, stream: cuda.CUstream):
        t = cute.make_tensor(ptr, cute.make_layout(shape, stride=strides))
        blocks = (cute.size(t) + THREADS - 1) // THREADS
        _kernel(t, value).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    def pointer(address):
        return make_ptr(element, address, cute.AddressSpace.gmem, assumed_align=align)

    compiled = cute.compile(fill, pointer(0), element(0), cuda.CUstream(0))

    def launch(address, value, stream):
        compiled(pointer(address), element(value), cuda.CUstream(stream))

    return launch


register("lumen::fill_", "cuda", compile)
