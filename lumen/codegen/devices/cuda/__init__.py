"""Python side of the CUDA backend: compiles generated CuTe DSL kernels for Rust to launch."""

import builtins
import linecache
import tempfile

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

# Alignment, in bytes, of the vectors kernels move their tensors by.
VECTOR_BYTES = 16

# The loaded modules of the compiled kernels, kept so their functions stay loaded.
modules = []


def compile(name, source, dtypes, aligned, threads, step_bytes):
    """Compiles the `launch` of the generated module `source` named `name` for tensors of CuTe types
    `dtypes` (``"cutlass.Float32"``…, passed by pointer: its inputs, then its outputs), moving them by
    16-byte vectors when `aligned` (every one starts 16-byte aligned), in blocks of `threads` threads each
    moving `step_bytes` of its widest tensor a step, and loads it into the current
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
    compiled, cubin = compiled_cubin(scope["launch"], *pointers, 1, cuda.CUstream(0), aligned, threads, step_bytes)
    (entry,) = compiled.kernel_info

    status, module = cuda.cuModuleLoadData(cubin)
    check(status, name)
    modules.append(module)
    status, function = cuda.cuModuleGetFunction(module, entry.encode())
    check(status, name)

    return int(function), threads // 32, 0, True, ["*"] * len(dtypes)


def compiled_cubin(f, *args):
    """`f` compiled by ``cute.compile`` for `args`, and its cubin's bytes. CuTe writes the cubin to a file named after
    the kernel, and lumen's kernels share names (every GEMM's is the same), so each compile dumps into a directory of
    its own: in a shared one, processes compiling at once read each other's cubins, half-written or of another
    kernel."""
    with tempfile.TemporaryDirectory() as dump:
        compiled = cute.compile(f, *args, options=f"--dump-dir={dump} --keep-cubin")
        return compiled, compiled.artifacts.CUBIN


def check(status, name):
    """Raises if CUDA driver call `status` failed loading kernel `name`."""
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{name}: loading its CuTe kernel failed ({status.name})")
