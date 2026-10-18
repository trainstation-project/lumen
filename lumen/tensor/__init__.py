"""Tensors: dtype constants, factory functions and NumPy interop over the
native ``lumen._C.Tensor`` (bindings in ``lumen/tensor/python.rs``).

Everything here is re-exported at the top level (``lumen.zeros``, ...).
Note that ``lumen.tensor`` itself is the factory function, as in PyTorch;
import this package's names from ``lumen`` (or ``from lumen.tensor import
...``).
"""

import builtins

from lumen._C import Tensor
from lumen.tensor import dlpack
from lumen.tensor.dlpack import from_dlpack

__all__ = [
    "Tensor",
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
    float16,
    bfloat16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
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
#
# Inside a function traced by `lumen.compile`, zeros/ones/full/arange
# record a `full` or `iota` primitive instead of allocating (as
# `torch.compile` does with `torch.zeros`). Graph values have no device,
# so `device` is ignored there: results land on the inputs' device.


def _prims():
    """`lumen.prims` while a function is being traced, else None."""
    from lumen.graph import prims, tracer  # lazily: lumen.graph imports this package

    return prims if tracer._TRACES else None


def tensor(data, dtype=None, device=None):
    """Build a tensor from (nested) lists. Dtype is inferred unless given."""
    return Tensor(data, dtype=dtype, device=device)


def empty(shape, dtype=None, device=None):
    """An uninitialized tensor, like ``torch.empty``: write every element
    (``fill_``, ``t[i] = v``, ...) before reading it. Reading first is
    undefined behavior in lumen's Rust core, not just garbage."""
    return Tensor.empty(list(shape), dtype, device)


def zeros(shape, dtype=None, device=None):
    if prims := _prims():
        return prims.full(shape, 0, dtype or default_dtype)
    return Tensor.zeros(list(shape), dtype, device)


def ones(shape, dtype=None, device=None):
    if prims := _prims():
        return prims.full(shape, 1, dtype or default_dtype)
    return Tensor.ones(list(shape), dtype, device)


def full(shape, value, dtype=None, device=None):
    if prims := _prims():
        if getattr(value, "shape", None) == ():
            # A traced scalar, broadcast (converted to ``dtype``).
            value = value if dtype in (None, value.dtype) else prims.cast(value, dtype)
            return prims.broadcast_in_dim(value, shape, ())
        inferred = "bool" if isinstance(value, builtins.bool) else "int64" if isinstance(value, int) else default_dtype
        return prims.full(shape, value, dtype or inferred)
    return Tensor.full(list(shape), value, dtype, device)


def arange(n, dtype=None, device=None):
    if prims := _prims():
        return prims.iota(dtype or default_dtype, (n,), 0)
    return Tensor.arange(n, dtype, device)


# ---------------------------------------------------------------------
# NumPy interop
# ---------------------------------------------------------------------


# Dtypes numpy has through ml_dtypes alone (each element a byte), copied
# through Python numbers.
_ML_DTYPES = ("int2", "int4", "uint2", "uint4", "float8_e4m3fn", "float8_e5m2", "float8_e4m3b11fnuz")


def to_numpy(t):
    """Copy a lumen.Tensor into a numpy array: one copy of its memory to the
    host (``Tensor.cpu``), read by numpy through DLPack, no Python number
    an element. numpy has no bfloat16: convert those first."""
    import numpy as np

    if t.dtype == "bfloat16":
        raise TypeError("numpy has no bfloat16: convert the tensor first (t.to(dtype=lumen.float32))")
    if t.dtype in _ML_DTYPES:
        import ml_dtypes

        return np.asarray(t.tolist(), dtype=getattr(ml_dtypes, t.dtype))
    host = t.cpu()
    array = np.from_dlpack(host)
    # A host tensor is its own memory: copy it, so the array is the caller's.
    return array.copy() if host.storage_id == t.storage_id else array


def from_numpy(a):
    """Copy a numpy array into a lumen.Tensor."""
    import numpy as np

    a = np.ascontiguousarray(a)
    values = a.ravel()
    if str(a.dtype) in _ML_DTYPES:
        values = values.astype(np.float64 if "float" in str(a.dtype) else np.int64)
    return Tensor(values.tolist(), shape=list(a.shape), dtype=str(a.dtype))


def _tensor__array__(self, dtype=None, copy=None):
    arr = to_numpy(self)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


try:  # native classes are heap types, so this usually sticks; harmless if not
    Tensor.__array__ = _tensor__array__
    Tensor.__dlpack__ = dlpack.__dlpack__
    Tensor.__dlpack_device__ = dlpack.__dlpack_device__
except (AttributeError, TypeError):  # pragma: no cover
    pass
