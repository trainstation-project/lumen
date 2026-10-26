"""Tests of the Python shim over the native tensor: re-exports, dtypes, factories and NumPy interop."""

import lumen
from lumen import Tensor
from lumen.tensor import _C


def test_shim_reexports_native_tensor():
    """`lumen.Tensor` is the native tensor class, presented under the `lumen` module."""
    assert _C.__name__ == "lumen.tensor._C"
    assert lumen.Tensor is _C.Tensor
    assert Tensor.__module__ == "lumen"
    assert isinstance(lumen.__version__, str)


def test_dtype_constants():
    """Dtype constants are their names, and every dtype can build a tensor."""
    assert lumen.float32 == "float32"
    assert lumen.bfloat16 == "bfloat16"
    assert lumen.int64 == "int64"
    assert lumen.bool == "bool"
    assert set(lumen.dtypes) == {
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
    }
    for dt in lumen.dtypes:
        assert lumen.zeros([2], dtype=dt).dtype == dt


def test_factories():
    """Tensor factories produce the expected dtypes, shapes and values."""
    assert lumen.tensor([1, 2, 3]).dtype == "int64"
    assert lumen.zeros([2, 3]).tolist() == [[0.0] * 3] * 2
    assert lumen.ones([2, 2]).tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert lumen.full([2], 7, dtype=lumen.int32).tolist() == [7, 7]
    assert lumen.arange(4).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert lumen.zeros((2, 2)).shape == [2, 2]


def test_numpy_roundtrip():
    """Tensors convert to NumPy arrays and back, including 0-d arrays."""
    np = __import__("numpy")
    t = lumen.arange(6).reshape([2, 3])
    a = lumen.to_numpy(t)
    assert a.shape == (2, 3)
    assert a.dtype == np.float32
    assert a.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]

    b = lumen.from_numpy(a)
    assert b.shape == [2, 3]
    assert b.dtype == "float32"
    assert b.tolist() == t.tolist()
    s = lumen.from_numpy(np.array(7, dtype=np.int32))
    assert s.shape == [] and s.item() == 7


def test_np_array_protocol():
    """`np.array` accepts a tensor through the array protocol."""
    np = __import__("numpy")
    t = lumen.tensor([1, 2, 3])
    a = np.array(t)
    assert a.dtype == np.int64
    assert a.tolist() == [1, 2, 3]


def test_from_numpy_dtypes():
    """Every NumPy dtype maps to its lumen dtype."""
    np = __import__("numpy")
    for np_dtype, lumen_dtype in [
        (np.float16, "float16"),
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int8, "int8"),
        (np.int16, "int16"),
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.uint8, "uint8"),
        (np.uint16, "uint16"),
        (np.uint32, "uint32"),
        (np.uint64, "uint64"),
        (np.bool_, "bool"),
    ]:
        t = lumen.from_numpy(np.zeros((2,), dtype=np_dtype))
        assert t.dtype == lumen_dtype, (np_dtype, t.dtype)


def test_shim_tensor_keeps_storage_semantics():
    """Indexing returns a view sharing storage, so writes show in the base tensor."""
    t = lumen.arange(6).reshape([2, 3])
    row = t[1]
    row[0] = -1.0
    assert t[1, 0] == -1.0
    assert row.shares_storage_with(t)
