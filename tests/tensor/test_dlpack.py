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


class _DLDevice(ctypes.Structure):
    """``DLDevice`` from ``dlpack.h``."""

    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    """``DLDataType`` from ``dlpack.h``."""

    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    """``DLTensor`` from ``dlpack.h``."""

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    """Legacy ``DLManagedTensor``, with no deleter."""

    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


class _DLManagedTensorVersioned(ctypes.Structure):
    """DLPack 1.0 ``DLManagedTensorVersioned``, with no deleter."""

    _fields_ = [
        ("major", ctypes.c_uint32),
        ("minor", ctypes.c_uint32),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", _DLTensor),
    ]


# Hand-built capsules' structs and buffers, kept alive for as long as an imported tensor may read them.
_KEEP = []


def _capsule(data, shape, strides=None, byte_offset=0, device=(1, 0), dtype=(2, 32, 1), version=None):
    """A capsule around a hand-built DLPack tensor: ``data`` is a ctypes buffer or ``None``;
    ``version`` set to ``(major, minor)`` makes a versioned capsule."""
    n = len(shape)
    dl = _DLTensor(
        data=ctypes.cast(data, ctypes.c_void_p) if data is not None else None,
        device=_DLDevice(*device),
        ndim=n,
        dtype=_DLDataType(*dtype),
        shape=(ctypes.c_int64 * n)(*shape),
        strides=(ctypes.c_int64 * n)(*strides) if strides is not None else None,
        byte_offset=byte_offset,
    )
    if version is None:
        managed, name = _DLManagedTensor(dl_tensor=dl), b"dltensor"
    else:
        managed = _DLManagedTensorVersioned(major=version[0], minor=version[1], dl_tensor=dl)
        name = b"dltensor_versioned"
    _KEEP.append((data, dl.shape, dl.strides, managed))
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ctypes.addressof(managed), name, None)


def test_import_a_hand_built_capsule_with_byte_offset_and_strides():
    """A capsule giving explicit strides and an element-aligned byte offset imports that view."""
    buf = (ctypes.c_float * 6)(0, 1, 2, 3, 4, 5)
    t = lumen.from_dlpack(_capsule(buf, [2, 2], strides=[2, 1], byte_offset=8, version=(1, 0)))
    assert t.tolist() == [[2.0, 3.0], [4.0, 5.0]]


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"version": (2, 0)}, BufferError, "DLPack version 2.0 is not supported"),
        ({"byte_offset": 4}, BufferError, "without strides must have byte_offset 0"),
        ({"strides": [1], "byte_offset": 2}, BufferError, "byte_offset 2 is not a multiple of the element size 4"),
        ({"shape": [-1]}, BufferError, "negative DLPack size -1"),
        ({"strides": [-1]}, BufferError, "negative DLPack stride -1"),
        ({"device": (4, 0)}, BufferError, "unsupported DLPack device type 4"),
        ({"dtype": (2, 32, 4)}, BufferError, r"unsupported DLPack dtype \(code 2, 32 bits, 4 lanes\)"),
        ({"dtype": (5, 64, 1)}, BufferError, r"unsupported DLPack dtype \(code 5, 64 bits, 1 lanes\)"),
        ({"null": True}, BufferError, "null data pointer"),
    ],
)
def test_import_rejects_invalid_capsules(kwargs, error, match):
    """Each invalid capsule field is reported, and the capsule is left unconsumed."""
    buf = None if kwargs.pop("null", False) else (ctypes.c_float * 4)()
    kwargs.setdefault("shape", [2])
    cap = _capsule(buf, **kwargs)
    with pytest.raises(error, match=match):
        lumen.from_dlpack(cap)
    assert _capsule_name(cap) in ("dltensor", "dltensor_versioned")


def test_import_an_empty_tensor_with_no_data():
    """A tensor with a zero-sized dimension imports from a null data pointer."""
    t = lumen.from_dlpack(_capsule(None, [2, 0]))
    assert t.shape == [2, 0]
    assert t.tolist() == [[], []]


def test_round_trip_bfloat16():
    """bfloat16 exports with DLPack's bfloat code and imports back as bfloat16."""
    t = lumen.full([3], 1.5, dtype=lumen.bfloat16)
    u = lumen.from_dlpack(t.__dlpack__())
    assert u.dtype == "bfloat16"
    assert u.tolist() == [1.5, 1.5, 1.5]


@pytest.mark.parametrize("dtype", ["int4", "float8_e4m3fn"])
def test_narrow_dtypes_export_as_bytes(dtype):
    """Dtypes DLPack has no code for export as unsigned bytes of their storage."""
    t = lumen.zeros([3], dtype=dtype)
    u = lumen.from_dlpack(t.__dlpack__())
    assert u.dtype == "uint8"
    assert u.data_ptr() == t.data_ptr()
    assert u.tolist() == [0, 0, 0]


@pytest.mark.mps
def test_mps_export_to_the_cpu_copies():
    """An MPS tensor exported for the CPU (``dl_device=(kDLCPU, 0)``) is a
    copy there, with the tensor's values."""
    try:
        t = lumen.arange(4, dtype=lumen.float32, device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    assert lumen.from_dlpack(t.__dlpack__(dl_device=(1, 0))).device == "cpu:0"
    assert np.from_dlpack(t, device="cpu").tolist() == [0.0, 1.0, 2.0, 3.0]


def test_export_to_an_unknown_device_type_raises():
    """A DLPack device type lumen has no device for is a ``BufferError``."""
    with pytest.raises(BufferError, match="unsupported DLPack device type 99"):
        lumen.zeros([1]).__dlpack__(dl_device=(99, 0))


def test_import_from_a_producer_without_max_version():
    """A producer predating DLPack 1.0, whose ``__dlpack__`` takes no
    ``max_version``, is asked again without it."""

    class Legacy:
        """Wraps a tensor, exporting it the pre-1.0 way."""

        def __init__(self, t):
            """Keeps ``t`` to export."""
            self.t = t

        def __dlpack_device__(self):
            """The wrapped tensor's device."""
            return self.t.__dlpack_device__()

        def __dlpack__(self, stream=None):
            """A legacy capsule of the wrapped tensor."""
            return self.t.__dlpack__(stream=stream)

    t = lumen.full([3], 2.5, dtype=lumen.float32)
    assert lumen.from_dlpack(Legacy(t)).tolist() == [2.5, 2.5, 2.5]
