"""CUDA ``fill_``: a CuTe DSL kernel, registered for ``lumen::fill_`` on
CUDA when lumen is imported (see ``lumen.ops``).

The tensor reaches the kernel through DLPack as a CuTe tensor over lumen's
buffer, shape and strides included, so any view is filled in place: each
thread takes one logical element and CuTe's layout maps it to its address.
"""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from lumen.ops import register

THREADS = 256


@cute.kernel
def _kernel(t: cute.Tensor, value):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * THREADS + tidx
    if i < cute.size(t):
        t[i] = value


@cute.jit
def _fill(t: cute.Tensor, value, stream: cuda.CUstream):
    blocks = (cute.size(t) + THREADS - 1) // THREADS
    _kernel(t, value).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)


_compiled = {}


def fill(t, value):
    # Order does not matter to a fill, so put the smallest stride first: a
    # flat index into a CuTe layout walks its first mode fastest, so
    # neighboring threads then store to n
    # eighboring addresses.
    t = t.permute(sorted(range(t.ndim), key=lambda d: t.strides[d]))
    ct = from_dlpack(t)
    if t.dtype == "bool":
        value = bool(value)
    elif not t.dtype.startswith(("float", "bfloat")):
        value = int(value)
    value = ct.element_type(value)
    # The legacy default stream, which lumen's other CUDA work runs on.
    stream = cuda.CUstream(0)
    key = (t.dtype, tuple(t.shape), tuple(t.strides))
    if key not in _compiled:
        _compiled[key] = cute.compile(_fill, ct, value, stream)
    _compiled[key](ct, value, stream)


register("lumen::fill_", "cuda", fill)
