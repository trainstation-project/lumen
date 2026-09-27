"""lumen: a minimal tensor library prototype.

This package is a pure-Python shim over the native Rust extension
`lumen._C` — the same relationship `torch` has to `torch._C` and `jax`
to `jaxlib`. The native module defines the `Tensor` class and all
storage semantics, plus:

* ``lumen.device`` — a device, modeled on ``torch.device``.
* ``lumen.config`` — process-wide runtime settings, e.g.
  ``lumen.config.memory_caching = False`` to bypass the caching allocator
  (PyTorch: ``PYTORCH_NO_CUDA_MEMORY_CACHING=1``).

This shim adds:

* dtype constants (`lumen.float32`, ...)
* top-level factory functions (`lumen.zeros`, `lumen.arange`, ...) à la PyTorch
* NumPy interop (`lumen.to_numpy`, `lumen.from_numpy`, `np.array(t)`)
"""

from lumen._C import Tensor, __version__, config, device

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
    # factories
    "tensor", "zeros", "ones", "full", "arange",
    # numpy interop
    "to_numpy", "from_numpy",
]

# ---------------------------------------------------------------------
# dtype constants (PyTorch: torch.float32, ...)
# ---------------------------------------------------------------------

float16 = "float16"
bfloat16 = "bfloat16"
float32 = "float32"
float64 = "float64"
int8 = "int8"
int16 = "int16"
int32 = "int32"
int64 = "int64"
uint8 = "uint8"
uint16 = "uint16"
uint32 = "uint32"
uint64 = "uint64"
bool = "bool"  # shadows the builtin within this namespace, like torch.bool

dtypes = (
    float16, bfloat16, float32, float64,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    bool,
)

#: dtype used when none is given and none can be inferred (PyTorch:
#: torch.get_default_dtype() == float32)
default_dtype = float32


# ---------------------------------------------------------------------
# factory functions (PyTorch: torch.zeros(...), torch.tensor(...), ...)
# ---------------------------------------------------------------------

# `device` is a string ("cpu", "mps", "cuda:0") or a `lumen.device`;
# None means the CPU, as in PyTorch.

def tensor(data, dtype=None, device=None):
    """Build a tensor from (nested) lists. Dtype is inferred unless given."""
    return Tensor(data, dtype=dtype, device=device)


def zeros(shape, dtype=None, device=None):
    return Tensor.zeros(list(shape), dtype, device)


def ones(shape, dtype=None, device=None):
    return Tensor.full(list(shape), 1, dtype, device)


def full(shape, value, dtype=None, device=None):
    return Tensor.full(list(shape), value, dtype, device)


def arange(n, dtype=None, device=None):
    return Tensor.arange(n, dtype, device)


# ---------------------------------------------------------------------
# NumPy interop
# ---------------------------------------------------------------------

def to_numpy(t):
    """Copy a lumen.Tensor into a numpy array (via tolist())."""
    import numpy as np

    return np.array(t.tolist(), dtype=np.dtype(t.dtype)).reshape(t.shape)


def from_numpy(a):
    """Copy a numpy array into a lumen.Tensor."""
    import numpy as np

    a = np.ascontiguousarray(a)
    return Tensor(a.ravel().tolist(), shape=list(a.shape), dtype=str(a.dtype))


def _tensor__array__(self, dtype=None, copy=None):
    arr = to_numpy(self)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


try:  # native classes are heap types, so this usually sticks; harmless if not
    Tensor.__array__ = _tensor__array__
except (AttributeError, TypeError):  # pragma: no cover
    pass
