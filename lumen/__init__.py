"""lumen: a minimal tensor library prototype.

This package is a pure-Python shim over the native Rust extension
`lumen._C` — the same relationship `torch` has to `torch._C` and `jax`
to `jaxlib`. Each Python package mirrors the Rust folder whose bindings it
wraps:

* ``lumen.tensor`` (``lumen/tensor/``) — ``Tensor``, dtype constants,
  factory functions (``lumen.zeros``, ``lumen.arange``, ...) à la PyTorch,
  and NumPy interop (``lumen.to_numpy``, ``lumen.from_numpy``,
  ``np.array(t)``).
* ``lumen.allocator`` (``lumen/allocator/``) — ``lumen.config``, runtime
  settings such as ``lumen.config.static_allocator_bytes``.
* ``lumen.device`` — a device, modeled on ``torch.device``.
* ``lumen.profiler`` (``lumen/profiler/``) — the profiler, modeled on
  ``torch.profiler``.
* ``lumen.graph`` (``lumen/graph/``) — compiled execution:
  ``lumen.compile`` traces a function into a graph of primitive ops
  (``lumen.prims``, modeled on ``jax.lax``); traced tensors have PyTorch's
  operators (``x @ w``), layout and dtype casts.
* ``lumen.functional`` (``import lumen.functional as F``) — the ops, as
  functions, as ``torch`` and ``torch.nn.functional`` have them
  (``F.softmax(x, -1)``, ``F.sum``, ``F.where``, ``F.rms_norm``, ...).
* ``lumen.ops`` (``lumen/ops.py``) — registering device kernels written in
  Python (``lumen.ops.register``), e.g. CUDA kernels authored with CuTe DSL.
* ``lumen.safetensors`` (``lumen/safetensors/``) — ``save_file``,
  ``load_file`` and ``safe_open`` for safetensors files, as in
  ``safetensors.torch``.
* ``lumen.mps`` and ``lumen.cuda`` (``lumen/stream/``) — the device streams'
  ``synchronize``, modeled on ``torch.mps`` and ``torch.cuda``.

Everything is re-exported here. As in PyTorch, ``lumen.tensor`` is the
factory function, which takes precedence over the package of that name.
"""

from lumen import autograd, functional, graph, nn, ops, profiler, safetensors
from lumen._C import __version__, device
from lumen.allocator import config
from lumen.autograd import grad, jvp, value_and_grad, vjp
from lumen.graph import compile, make_graph, prims
from lumen.stream import cuda, mps
from lumen.tensor import tensor  # rebinds `lumen.tensor` from the package to the factory
from lumen.tensor import (
    Tensor,
    arange,
    bfloat16,
    bool,
    default_dtype,
    dtypes,
    empty,
    float16,
    float32,
    float64,
    from_dlpack,
    from_numpy,
    full,
    int8,
    int16,
    int32,
    int64,
    ones,
    to_numpy,
    uint8,
    uint16,
    uint32,
    uint64,
    zeros,
)

__all__ = [
    "Tensor",
    "__version__",
    "config",
    "cuda",
    "device",
    "graph",
    "mps",
    "nn",
    "ops",
    "profiler",
    "safetensors",
    # dtypes
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "dtypes",
    "default_dtype",
    # factories
    "tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "arange",
    # numpy interop
    "to_numpy",
    "from_numpy",
    "from_dlpack",
    # compiled execution
    "compile",
    "grad",
    "jvp",
    "value_and_grad",
    "vjp",
    "make_graph",
    "prims",
    "functional",
    "autograd",
]
