"""Python side of the CUDA backend: compiles generated CuTe DSL kernels for Rust to launch."""

import builtins
import linecache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from lumen.codegen.elementwise.cuda.kernel import THREADS

# Alignment, in bytes, of the vectors kernels move their tensors by.
VECTOR_BYTES = 16

# The loaded modules of the compiled kernels, kept so their functions stay loaded.
modules = []


def compile(name, source, dtypes, aligned):
    """Compiles the `launch` of the generated module `source` named `name` for tensors of CuTe types
    `dtypes` (``"cutlass.Float32"``…, passed by pointer: its inputs, then its outputs), moving them by
    16-byte vectors when `aligned` (every one starts 16-byte aligned), and loads it into the current
    context. Returns it as a Triton kernel's launch info: its ``CUfunction``, warps per block, dynamic
    shared bytes, whether a plain ``cuLaunchKernel`` runs it, and its parameters' types (each a pointer)."""
    filename = f"<lumen {name}>"
    # CuTe reads its functions' source, so register the generated module's with linecache.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope = {"__name__": name}
    exec(builtins.compile(source, filename, "exec"), scope)

    align = VECTOR_BYTES if aligned else None
    types = [getattr(cutlass, d.removeprefix("cutlass.")) for d in dtypes]
    pointers = [make_ptr(t, VECTOR_BYTES, cute.AddressSpace.gmem, assumed_align=align) for t in types]
    compiled = cute.compile(scope["launch"], *pointers, 1, cuda.CUstream(0), aligned, options="--keep-cubin")
    (entry,) = compiled.kernel_info

    status, module = cuda.cuModuleLoadData(compiled.artifacts.CUBIN)
    check(status, name)
    modules.append(module)
    status, function = cuda.cuModuleGetFunction(module, entry.encode())
    check(status, name)

    return int(function), THREADS // 32, 0, True, ["*"] * len(dtypes)


def check(status, name):
    """Raises if CUDA driver call `status` failed loading kernel `name`."""
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name}: loading its CuTe kernel failed ({status.name})")
