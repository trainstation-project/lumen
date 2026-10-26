"""Python tensor API: dtype names, tensor factories (traced as prims inside ``lumen.compile``) and numpy/DLPack interop."""

import builtins

from lumen.tensor import dlpack
from lumen.tensor._C import Tensor
from lumen.tensor.dlpack import from_dlpack

__all__ = [
    "Tensor",
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
    "tensor",
    "empty",
    "zeros",
    "ones",
    "full",
    "arange",
    "to_numpy",
    "from_numpy",
    "from_dlpack",
]


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
bool = "bool"

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

default_dtype = float32


def _prims():
    """`lumen.prims` while a function is being traced, else None."""
    from lumen.graph import prims, tracer

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
    """A tensor of zeros; inside a trace, a ``full`` prim in ``dtype`` or the default dtype."""
    if prims := _prims():
        return prims.full(shape, 0, dtype or default_dtype)
    return Tensor.zeros(list(shape), dtype, device)


def ones(shape, dtype=None, device=None):
    """A tensor of ones; inside a trace, a ``full`` prim in ``dtype`` or the default dtype."""
    if prims := _prims():
        return prims.full(shape, 1, dtype or default_dtype)
    return Tensor.ones(list(shape), dtype, device)


def full(shape, value, dtype=None, device=None):
    """A tensor filled with ``value``; inside a trace, a ``full`` prim, or a broadcast when ``value`` is a traced scalar.

    Without ``dtype``, a traced fill takes bool, int64 or the default dtype from ``value``'s Python type."""
    if prims := _prims():
        if getattr(value, "shape", None) == ():
            value = value if dtype in (None, value.dtype) else prims.cast(value, dtype)
            return prims.broadcast_in_dim(value, shape, ())
        inferred = "bool" if isinstance(value, builtins.bool) else "int64" if isinstance(value, int) else default_dtype
        return prims.full(shape, value, dtype or inferred)
    return Tensor.full(list(shape), value, dtype, device)


def arange(n, dtype=None, device=None):
    """``0, 1, ..., n - 1``; inside a trace, an ``iota`` prim."""
    if prims := _prims():
        return prims.iota(dtype or default_dtype, (n,), 0)
    return Tensor.arange(n, dtype, device)


# Dtypes numpy reads through the ``ml_dtypes`` package, converted element by element.
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
    # ``cpu()`` returns the same storage for a host tensor, so copy to keep the array independent.
    return array.copy() if host.storage_id == t.storage_id else array


def from_numpy(a):
    """Copy a numpy array into a lumen.Tensor."""
    import numpy as np

    shape = np.shape(a)
    a = np.ascontiguousarray(a)
    values = a.ravel()
    if str(a.dtype) in _ML_DTYPES:
        values = values.astype(np.float64 if "float" in str(a.dtype) else np.int64)
    return Tensor(values.tolist(), shape=list(shape), dtype=str(a.dtype))


def _tensor__array__(self, dtype=None, copy=None):
    """numpy's ``__array__`` protocol for ``Tensor``, casting to ``dtype`` when given."""
    arr = to_numpy(self)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


# Attach numpy and DLPack protocols to the native Tensor class.
try:
    Tensor.__array__ = _tensor__array__
    Tensor.__dlpack__ = dlpack.__dlpack__
    Tensor.__dlpack_device__ = dlpack.__dlpack_device__
except (AttributeError, TypeError):  # pragma: no cover
    pass
