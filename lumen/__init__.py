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
  settings such as ``lumen.config.memory_caching``.
* ``lumen.device`` — a device, modeled on ``torch.device``.

Everything is re-exported here. As in PyTorch, ``lumen.tensor`` is the
factory function, which takes precedence over the package of that name.
"""

from lumen._C import __version__, device
from lumen.allocator import config
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
    from_numpy,
    full,
    int8,
    int16,
    int32,
    int64,
    ones,
    tensor,  # rebinds `lumen.tensor` from the package to the factory
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
    "device",
    # dtypes
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
    "dtypes",
    "default_dtype",
    # factories
    "tensor", "empty", "zeros", "ones", "full", "arange",
    # numpy interop
    "to_numpy", "from_numpy",
]
