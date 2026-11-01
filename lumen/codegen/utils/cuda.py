"""Python side of the CUDA backend: compiles generated CuTe DSL kernels and launches them."""

import linecache

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

kernels = {}


def compiled(name, source, tensors):
    """The `launch` of the generated module `source` named `name`, compiled through TVM FFI for tensors like
    `tensors` (flat views: one compiled kernel serves every later set of them) on its first use."""
    if name not in kernels:
        filename = f"<lumen {name}>"
        # CuTe reads its functions' source, so register the generated module's with linecache.
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        scope = {"__name__": name}
        exec(compile(source, filename, "exec"), scope)
        examples = [from_dlpack(t) for t in tensors]
        kernels[name] = cute.compile(scope["launch"], *examples, 1, cuda.CUstream(0), options="--enable-tvm-ffi")
    return kernels[name]


def launch(name, source, tensors, blocks, stream):
    """Launches kernel `name` of `source` on `tensors` (its inputs, then its outputs) on `blocks` blocks on
    CUDA stream `stream` (a `cudaStream_t` integer, 0 for the default stream)."""
    compiled(name, source, tensors)(*tensors, blocks, cuda.CUstream(stream))
