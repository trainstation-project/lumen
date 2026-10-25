import linecache
import math
import struct

import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 1024
kernels = {}


def compiled(name, source):
    if name not in kernels:
        filename = f"<lumen {name}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        scope = {"__name__": __name__, "triton": triton, "tl": tl, "libdevice": libdevice}
        exec(compile(source, filename, "exec"), scope)
        kernels[name] = scope[name]
    return kernels[name]


def functions(specs):
    return {param: compiled(name, source) for param, name, source in specs}


def scalar(x):
    """A runtime scalar as a kernel argument: a float64's bits as an int64
    (a Python float argument is typed float32), any other's value."""
    if x.dtype == "float64":
        return struct.unpack("<q", struct.pack("<d", x.item()))[0]
    return x.item()


def launch(name, source, inputs, scalars, out, specs):
    n = math.prod(out.shape)
    if n == 0:
        return
    args = [scalar(x) if k in scalars else x for k, x in enumerate(inputs)]
    kernel = compiled(name, source)
    kernel[(triton.cdiv(n, BLOCK),)](*args, out, n, **functions(specs), BLOCK=BLOCK, enable_fp_fusion=False)
