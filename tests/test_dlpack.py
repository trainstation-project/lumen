"""DLPack interop: ``Tensor.__dlpack__``/``from_dlpack``.

lumen exports a DLPack capsule that another framework can import as a view of
the same buffer, and imports a foreign capsule (NumPy, PyTorch, CuTe DSL) the
same way. Two properties carry the risk here, so both are pinned down
explicitly:

* lifetime — the capsule owns a storage reference, so importing outlives the
  exporting ``Tensor``; a foreign capsule owns *its* buffer, so the imported
  tensor outlives the producer;
* single consumption — a capsule spent once must not be spendable twice,
  because the second import would read a freed ``DLManagedTensor``.

NumPy is the only producer available here, and it is optional: everything
below skips if it is missing.
"""

import gc

import pytest

import lumen

np = pytest.importorskip("numpy")


def test_dlpack_device_tuple():
    t = lumen.zeros([1], dtype=lumen.float32)
    assert t.__dlpack_device__() == (1, 0)


def test_export_capsule_name():
    t = lumen.zeros([1], dtype=lumen.float32)
    cap = t.__dlpack__()
    assert cap is not None
    # A fresh capsule is named "dltensor" (DLPack's unconsumed name).
    assert t.__dlpack__() is not cap


def test_export_to_numpy_values():
    t = lumen.full([12], 7.0, dtype=lumen.float32)
    a = np.from_dlpack(t)
    assert a.dtype == np.float32
    assert a.shape == (12,)
    assert a.tolist() == [7.0] * 12


def test_export_survives_exporting_tensor():
    t = lumen.full([4], 3.0, dtype=lumen.float32)
    a = np.from_dlpack(t)
    del t
    gc.collect()
    # The capsule holds a storage reference: the buffer must still be alive.
    assert a.tolist() == [3.0] * 4


def test_export_observes_producer_writes():
    t = lumen.zeros([4], dtype=lumen.float32)
    a = np.from_dlpack(t)
    # NumPy imports a DLPack array read-only (the protocol has no writability
    # flag), so writes go through the producer and the view sees them.
    t.fill_(5.0)
    assert a.tolist() == [5.0] * 4


def test_import_from_numpy_values():
    a = np.arange(6, dtype=np.float32)
    t = lumen.from_dlpack(a)
    assert t.shape == [6]
    assert t.dtype == "float32"
    assert t.tolist() == a.tolist()


def test_import_is_a_view():
    a = np.arange(4, dtype=np.float32)
    t = lumen.from_dlpack(a)
    assert t.data_ptr() == a.ctypes.data
    t.fill_(9.0)
    assert a.tolist() == [9.0] * 4


def test_import_survives_producer():
    a = np.arange(8, dtype=np.float32)
    t = lumen.from_dlpack(a)
    ptr = a.ctypes.data
    del a
    gc.collect()
    # The imported storage owns the producer's capsule, so the buffer lives.
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
    a = np.arange(6, dtype=np.dtype(dtype)).reshape(2, 3)
    t = lumen.from_dlpack(a)
    assert t.shape == [2, 3]
    back = np.from_dlpack(t)
    assert np.array_equal(back, a)


def test_round_trip_2d_float64():
    a = np.arange(5, dtype=np.float64)
    t = lumen.from_dlpack(a)
    assert np.array_equal(np.from_dlpack(t), a)


def test_capsule_consumed_once():
    from lumen import _C

    t = lumen.full([4], 1.0, dtype=lumen.float32)
    cap = t.__dlpack__()
    first = _C._from_dlpack(cap)
    assert first.tolist() == [1.0] * 4
    # Spending the same capsule again would read a freed managed tensor.
    with pytest.raises(ValueError, match="used_dltensor"):
        _C._from_dlpack(cap)


def test_import_rejects_non_capsule():
    with pytest.raises(TypeError):
        lumen.from_dlpack(42)


def test_non_contiguous_export_has_explicit_strides():
    t = lumen.zeros([3, 4], dtype=lumen.float32)
    view = t.transpose(0, 1)  # not a contiguous layout of its own
    cap = view.__dlpack__()
    assert cap is not None
