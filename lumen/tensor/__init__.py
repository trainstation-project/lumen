"""Python tensor API: dtype names, tensor factories (traced as prims inside ``lumen.compile``), numpy/DLPack interop
and the methods of axis names."""

import builtins

from lumen.tensor import dlpack
from lumen.tensor import names as _names
from lumen.tensor import placements as _placements
from lumen.tensor._C import Tensor
from lumen.tensor.dlpack import from_dlpack
from lumen.tensor.names import AxisType

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
    from lumen.graph.tracing import prims, tracer

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

    if t.placement is not None:
        raise TypeError("to_numpy: the tensor is sharded: take this process's shard (to_local()) first")
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
    if str(a.dtype) not in _ML_DTYPES:
        # Copy the buffer through DLPack; arrays it can't export (read-only ones) go through a list.
        try:
            return from_dlpack(a.reshape(shape)).clone()
        except (BufferError, TypeError):
            pass
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


def _rename(self, *names, **mapping):
    """It, sharing its memory, with its axes named ``names``, one per dim (a
    lone None drops the names), or renamed by ``mapping`` (old name to
    new), as torch's ``rename``. Names are metadata a compiled function
    reads at trace time (``TracedTensor``): its results come back named."""
    return self._with_names(_names.renamed(self.names, self.ndim, names, mapping))


def _refine_names(self, *names):
    """It named ``names`` if it is unnamed; a named tensor must have them already (torch's ``refine_names``)."""
    return self._with_names(_names.refined(self.names, self.ndim, names))


def _align_to(self, *order):
    """A view of it with its axes in ``order`` (names; one ``...`` stands for the others in their order)."""
    order = _names.order(self.names, order, "align_to")
    if order == self.names:
        return self
    return self.permute([self.names.index(n) for n in order])._with_names(order)


def _size(self, dim: AxisType | None = None):
    """The shape, or the size of dimension ``dim`` (a position or an axis name)."""
    return self.shape if dim is None else self.shape[_names.axis(dim, self.ndim, self.names, "size")]


def _dim_of(self, name: str) -> int:
    """The position of axis ``name``; a KeyError if it has none."""
    return _names.position(self.names, name, "dim_of")


def _distribute(self, mesh, placements):
    """It, a global tensor every process holds, placed over ``mesh`` by ``placements`` (one ``Shard``,
    ``Replicate`` or ``Partial`` per mesh dimension; ``lumen/tensor/placements.py``): a copy of this
    process's shard (zeros for a partial sum's parts but the first; a meta tensor's is a new meta tensor),
    named as it is (DTensor's ``distribute_tensor``)."""
    if self.placement is not None:
        raise TypeError("distribute: the tensor is sharded already")
    spec = _placements.sharding(mesh, placements, self.ndim, self.names)
    local = _placements.local_shape(self.shape, spec)
    at = _placements.coordinate(mesh)
    parts = enumerate(spec.placements)
    zero = any(isinstance(p, _placements.Partial) and p.reduce_op == "sum" and at[i] for i, p in parts)
    if self.device.startswith("meta"):
        shard = Tensor.empty(list(local), self.dtype, self.device)
    elif zero:
        shard = Tensor.zeros(list(local), self.dtype, self.device)
    else:
        shard = self
        for d, start in enumerate(_placements.offsets(self.shape, spec)):
            if local[d] != self.shape[d]:
                shard = shard.narrow(d, start, local[d])
        shard = shard.clone()
    return shard._with_names(self.names)._with_placement(spec, list(self.shape))


def _to_local(self):
    """Its local shard, sharing its memory, a local tensor (DTensor's ``to_local``)."""
    return self if self.placement is None else self._with_placement(None)


def _placements_of(self):
    """Its placement on each mesh dimension (a tuple), or None for a local tensor (DTensor's ``placements``)."""
    return None if self.placement is None else self.placement.placements


def _mesh_of(self):
    """The mesh it is sharded over, or None for a local tensor (DTensor's ``device_mesh``)."""
    return None if self.placement is None else self.placement.mesh


def _traced_only(name):
    """A method that runs only on traced tensors, inside ``lumen.compile`` (it records collectives)."""

    def method(self, *args, **kwargs):
        """Raises: the method records collectives, which run inside ``lumen.compile``."""
        from lumen.graph.tracing import tracer

        tracer._lift(self)
        tracer.current_graph()

    method.__name__ = name
    return method


# Attach numpy and DLPack protocols, and the methods of names and placements, to the native Tensor class.
try:
    Tensor.__array__ = _tensor__array__
    Tensor.__dlpack__ = dlpack.__dlpack__
    Tensor.__dlpack_device__ = dlpack.__dlpack_device__
    Tensor.rename = _rename
    Tensor.refine_names = _refine_names
    Tensor.align_to = _align_to
    Tensor.size = _size
    Tensor.dim_of = _dim_of
    Tensor.distribute = _distribute
    Tensor.to_local = _to_local
    Tensor.full_tensor = _traced_only("full_tensor")
    Tensor.redistribute = _traced_only("redistribute")
    Tensor.placements = property(_placements_of)
    Tensor.device_mesh = property(_mesh_of)
except (AttributeError, TypeError):  # pragma: no cover
    pass
