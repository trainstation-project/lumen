"""Tests of DLPack export and import: capsule protocol, streams, devices, dtypes, strides and lifetimes."""

import ctypes
import gc

import pytest

import lumen
from lumen.tensor import _C

np = pytest.importorskip("numpy")


def _capsule_name(capsule):
    """Name of a PyCapsule, read through the CPython API."""
    get_name = ctypes.pythonapi.PyCapsule_GetName
    get_name.restype = ctypes.c_char_p
    get_name.argtypes = [ctypes.py_object]
    return get_name(capsule).decode()


def test_dlpack_device_tuple():
    """A CPU tensor reports DLPack device `(kDLCPU, 0)`."""
    t = lumen.zeros([1], dtype=lumen.float32)
    assert t.__dlpack_device__() == (1, 0)


def test_export_capsule_names_follow_max_version():
    """Capsules are unversioned unless the consumer accepts DLPack 1.0."""
    t = lumen.zeros([1], dtype=lumen.float32)
    assert _capsule_name(t.__dlpack__()) == "dltensor"
    assert _capsule_name(t.__dlpack__(max_version=(0, 8))) == "dltensor"
    assert _capsule_name(t.__dlpack__(max_version=(1, 0))) == "dltensor_versioned"


@pytest.mark.parametrize("max_version", [None, (1, 0)])
def test_import_renames_the_capsule(max_version):
    """Importing a capsule marks it used by renaming it."""
    t = lumen.full([2], 4.0, dtype=lumen.float32)
    cap = t.__dlpack__(max_version=max_version)
    name = _capsule_name(cap)
    assert lumen.from_dlpack(cap).tolist() == [4.0, 4.0]
    assert _capsule_name(cap) == "used_" + name


def test_stream_must_be_an_int_or_none():
    """A non-integer stream is a type error."""
    t = lumen.zeros([1])
    with pytest.raises(TypeError):
        t.__dlpack__(stream="default")


def test_cpu_export_takes_no_stream():
    """CPU export accepts only no stream or -1."""
    t = lumen.zeros([1])
    t.__dlpack__(stream=None)
    t.__dlpack__(stream=-1)
    with pytest.raises(ValueError, match="stream"):
        t.__dlpack__(stream=5)


def test_export_to_its_own_device():
    """Exporting to the tensor's own device without a copy works."""
    t = lumen.zeros([1])
    assert _capsule_name(t.__dlpack__(dl_device=(1, 0), copy=False)) == "dltensor"


def test_export_cannot_copy_on_the_same_device():
    """A forced copy on the same device is refused."""
    with pytest.raises(BufferError):
        lumen.zeros([1]).__dlpack__(copy=True)


def test_export_to_another_device_without_copying_raises():
    """Another device can't be reached without copying."""
    with pytest.raises(ValueError, match="without copying"):
        lumen.zeros([1]).__dlpack__(dl_device=(2, 0), copy=False)


def test_export_to_numpy_values():
    """NumPy reads an exported tensor's values, dtype and shape."""
    t = lumen.full([12], 7.0, dtype=lumen.float32)
    a = np.from_dlpack(t)
    assert a.dtype == np.float32
    assert a.shape == (12,)
    assert a.tolist() == [7.0] * 12


def test_export_survives_exporting_tensor():
    """An exported array stays valid after the tensor is freed."""
    t = lumen.full([4], 3.0, dtype=lumen.float32)
    a = np.from_dlpack(t)
    del t
    gc.collect()
    assert a.tolist() == [3.0] * 4


def test_export_observes_producer_writes():
    """An exported array aliases the tensor, seeing later writes."""
    t = lumen.zeros([4], dtype=lumen.float32)
    a = np.from_dlpack(t)
    t.fill_(5.0)
    assert a.tolist() == [5.0] * 4


def test_import_from_numpy_values():
    """Importing a NumPy array keeps its values, shape and dtype."""
    a = np.arange(6, dtype=np.float32)
    t = lumen.from_dlpack(a)
    assert t.shape == [6]
    assert t.dtype == "float32"
    assert t.tolist() == a.tolist()


def test_import_is_a_view():
    """An imported tensor aliases the array's memory."""
    a = np.arange(4, dtype=np.float32)
    t = lumen.from_dlpack(a)
    assert t.data_ptr() == a.ctypes.data
    t.fill_(9.0)
    assert a.tolist() == [9.0] * 4


def test_import_survives_producer():
    """An imported tensor keeps the array's memory alive after the array is freed."""
    a = np.arange(8, dtype=np.float32)
    t = lumen.from_dlpack(a)
    ptr = a.ctypes.data
    del a
    gc.collect()
    assert t.data_ptr() == ptr
    assert t.tolist() == [float(i) for i in range(8)]


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    ],
)
def test_round_trip_dtypes(dtype):
    """Integer and float arrays survive NumPy to lumen to NumPy."""
    a = np.arange(6, dtype=np.dtype(dtype)).reshape(2, 3)
    t = lumen.from_dlpack(a)
    assert t.shape == [2, 3]
    back = np.from_dlpack(t)
    assert np.array_equal(back, a)


def test_round_trip_2d_float64():
    """A float64 array survives the round trip."""
    a = np.arange(5, dtype=np.float64)
    t = lumen.from_dlpack(a)
    assert np.array_equal(np.from_dlpack(t), a)


def test_capsule_consumed_once():
    """A capsule can be imported only once."""
    t = lumen.full([4], 1.0, dtype=lumen.float32)
    cap = t.__dlpack__()
    first = _C._from_dlpack(cap)
    assert first.tolist() == [1.0] * 4
    with pytest.raises(RuntimeError, match="consumed only once"):
        _C._from_dlpack(cap)


def test_import_from_a_lumen_tensor_is_a_view():
    """Importing a lumen tensor gives a view of its storage."""
    t = lumen.arange(4, dtype=lumen.float32)
    u = lumen.from_dlpack(t)
    assert u.data_ptr() == t.data_ptr()
    u.fill_(2.0)
    assert t.tolist() == [2.0] * 4


def test_import_keeps_a_strided_layout():
    """A sliced array imports with its strides and data pointer, without a copy."""
    a = np.arange(12, dtype=np.float32).reshape(3, 4)[:, 1:3]
    t = lumen.from_dlpack(a)
    assert t.shape == [3, 2]
    assert t.strides == [4, 1]
    assert t.data_ptr() == a.ctypes.data
    assert t.tolist() == a.tolist()


def test_import_a_transposed_array():
    """A transposed array imports with transposed strides."""
    a = np.arange(6, dtype=np.int32).reshape(2, 3).T
    t = lumen.from_dlpack(a)
    assert t.strides == [1, 3]
    assert t.tolist() == a.tolist()


def test_import_a_0d_array():
    """A 0-d array round-trips."""
    t = lumen.from_dlpack(np.array(3.5))
    assert t.shape == []
    assert np.from_dlpack(t).item() == 3.5


def test_round_trip_bool():
    """Bool arrays round-trip."""
    a = np.array([True, False, True])
    t = lumen.from_dlpack(a)
    assert t.dtype == "bool"
    assert np.from_dlpack(t).tolist() == [True, False, True]


def test_import_rejects_non_capsule():
    """Importing something that isn't a capsule or DLPack producer is a type error."""
    with pytest.raises(TypeError):
        lumen.from_dlpack(42)


def test_export_a_strided_view():
    """A transposed, narrowed view exports with its strides."""
    t = lumen.arange(12, dtype=lumen.float32).reshape([3, 4])
    a = np.from_dlpack(t.transpose(0, 1).narrow(0, 1, 2))
    assert a.tolist() == [[1.0, 5.0, 9.0], [2.0, 6.0, 10.0]]


@pytest.mark.mps
def test_mps_dlpack_device():
    """An MPS tensor reports DLPack device `(kDLMetal, 0)` and imports back onto MPS."""
    try:
        t = lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    assert t.__dlpack_device__() == (8, 0)
    assert lumen.from_dlpack(t.narrow(0, 0, 1)).device == "mps:0"
