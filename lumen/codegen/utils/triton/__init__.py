"""Python side of the Triton backend: compiles generated kernel sources and launches them."""

import importlib
import linecache
import struct

import triton
import triton.language as tl
from triton.language.extra import libdevice

from lumen.codegen.random.triton.philox import philox_bits

kernels = {}


def compiled(name, source):
    """Returns the Triton kernel `name` defined by `source`, compiling and caching it on first use."""
    if name not in kernels:
        filename = f"<lumen {name}>"
        # Register the source with linecache so tracebacks and Triton's source inspection can see it.
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        scope = {
            "__name__": __name__,
            "triton": triton,
            "tl": tl,
            "libdevice": libdevice,
            "philox_bits": philox_bits,
        }
        exec(compile(source, filename, "exec"), scope)
        kernels[name] = scope[name]
    return kernels[name]


def functions(specs):
    """Maps each `(param, name, source)` spec to its compiled helper kernel, keyed by parameter name."""
    return {param: compiled(name, source) for param, name, source in specs}


def scalar(x):
    """A runtime scalar as a kernel argument: a float64's bits as an int64
    (a Python float argument is typed float32), any other's value."""
    if x.dtype == "float64":
        return struct.unpack("<q", struct.pack("<d", x.item()))[0]
    return x.item()


def compiled_launch(kernel, grid, args, kwargs):
    """`kernel` compiled for a launch with `args` and `kwargs` on `grid`, without launching it, for Rust to
    launch: its `CUfunction` handle, warps, dynamic shared memory, whether a plain `cuLaunchKernel` launches
    it (one CTA, no scratch, no cooperative or programmatic launch), and the type of each runtime parameter
    in declaration order (tuples flattened; ``constexpr`` where Triton specialized the value away)."""
    c = kernel.warmup(*args, grid=grid, **kwargs)
    c._init_handles()
    signature, types = c.src.signature, []

    def flat(t):
        """Appends `t`'s types, a tuple's elements in order."""
        if isinstance(t, tuple):
            for x in t:
                flat(x)
        else:
            types.append(t)

    for i, p in enumerate(kernel.params):
        if not p.is_constexpr:
            flat(signature[p.name] if p.name in signature else signature[i])
    md = c.metadata
    plain = (
        md.num_ctas == 1
        and not md.launch_cooperative_grid
        and not md.launch_pdl
        and not md.global_scratch_size
        and not md.profile_scratch_size
    )
    return int(c.function), md.num_warps, md.shared, plain, types


def template(module, kernel, tensors, scalars, specs, constexprs, programs, block_size, warps, compile=False):
    """Launches template kernel `kernel` of `module` on a 1-D grid of `programs` blocks, or, with `compile`,
    returns its `compiled_launch` instead: `tensors` reach it as its `EXTRA` tuple (those whose index is in
    `scalars` by value), `specs` supply its generated jit functions and `constexprs` its other constants."""
    if programs == 0:
        return None
    kernel = getattr(importlib.import_module(module), kernel)
    extra = tuple(scalar(x) if k in scalars else x for k, x in enumerate(tensors))
    kwargs = dict(**functions(specs), **constexprs, BLOCK_SIZE=block_size, num_warps=warps, enable_fp_fusion=False)
    if compile:
        return compiled_launch(kernel, (programs,), (extra,), kwargs)
    kernel[(programs,)](extra, **kwargs)


def launch(name, source, inputs, scalars, out, specs, n, programs, block_size, warps, compile=False):
    """Launches elementwise kernel `name` over `n` output elements on a 1-D grid of `programs` blocks, or,
    with `compile`, returns its `compiled_launch` instead.

    Inputs whose index is in `scalars` are passed by value; `specs` supplies helper jit functions."""
    if n == 0:
        return None
    args = [*(scalar(x) if k in scalars else x for k, x in enumerate(inputs)), out, n]
    kernel = compiled(name, source)
    kwargs = dict(**functions(specs), BLOCK_SIZE=block_size, num_warps=warps, enable_fp_fusion=False)
    if compile:
        return compiled_launch(kernel, (programs,), args, kwargs)
    kernel[(programs,)](*args, **kwargs)
