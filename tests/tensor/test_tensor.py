"""Tests of the host `Tensor`: construction, dtypes, factories, strided views sharing storage, indexing, conversion and numpy export."""

import numpy as np
import pytest

import lumen
from lumen import Tensor


def test_construct_and_infer_dtype():
    """The constructor infers float32, int64 and bool from Python data and places the tensor on the CPU."""
    t = Tensor([[0.0, 1.0], [2.0, 3.0]])
    assert t.shape == [2, 2]
    assert t.dtype == "float32"
    assert t.device == "cpu:0"

    t = Tensor([1, 2, 3])
    assert t.dtype == "int64"

    t = Tensor([True, False])
    assert t.dtype == "bool"


def test_explicit_dtype():
    """An explicit dtype converts the given values."""
    t = Tensor([1, 2, 3], dtype="float32")
    assert t.dtype == "float32"
    assert t.tolist() == [1.0, 2.0, 3.0]


def test_extended_dtypes():
    """Every integer width, float16 and bfloat16 store and read back values."""
    for dt in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
        t = Tensor([-1, 2] if dt.startswith("int") else [1, 2], dtype=dt)
        assert t.dtype == dt
        assert t[1] == 2
    t = Tensor([1.5, 2.5], dtype="float16")
    assert t.dtype == "float16"
    assert t.tolist() == [1.5, 2.5]
    t[0] = 3.25
    assert t[0] == 3.25
    t = Tensor([1.5], dtype="bfloat16")
    assert t.dtype == "bfloat16"
    assert t[0] == 1.5


def test_zeros_full_arange():
    """The `zeros`, `full` and `arange` factories, with float32 as arange's default dtype."""
    assert Tensor.zeros([2, 3]).tolist() == [[0.0] * 3] * 2
    assert Tensor.full([2, 2], 7).tolist() == [[7, 7], [7, 7]]
    assert Tensor.arange(5).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert Tensor.arange(3, dtype="int32").dtype == "int32"


def test_strides_row_major():
    """New tensors are contiguous with row-major strides."""
    t = Tensor.zeros([2, 3, 4])
    assert t.strides == [12, 4, 1]
    assert t.is_contiguous()


def test_reshape_shares_storage():
    """`reshape` of a contiguous tensor is a view of the same storage."""
    t = Tensor.arange(6)
    m = t.reshape([2, 3])
    assert t.shares_storage_with(m)
    assert t.storage_id == m.storage_id
    assert m[1, 2] == 5.0


def test_mutation_visible_through_aliases():
    """A write through a reshape view is visible through the original."""
    t = Tensor.arange(6)
    m = t.reshape([2, 3])
    m[0, 1] = 99.0
    assert t[1] == 99.0


def test_narrow_bumps_offset():
    """`narrow` returns a view offset into the same storage."""
    t = Tensor.arange(10)
    v = t.narrow(0, 3, 4)
    assert v.storage_offset == 3
    assert v.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert t.shares_storage_with(v)


def test_getitem_view_and_scalar():
    """Indexing with fewer indices than dimensions returns a view; with all of them, a scalar."""
    t = Tensor.arange(6).reshape([2, 3])
    row = t[1]
    assert isinstance(row, Tensor)
    assert row.tolist() == [3.0, 4.0, 5.0]
    assert row.shares_storage_with(t)
    assert t[1, 2] == 5.0
    assert t[-1].tolist() == [3.0, 4.0, 5.0]


def test_setitem():
    """Item assignment writes through the tensor and through row views, with negative indices."""
    t = Tensor.zeros([2, 2])
    t[1, 1] = 5.0
    assert t.tolist() == [[0.0, 0.0], [0.0, 5.0]]
    row = t[1]
    row[0] = 7.0
    assert t[1, 0] == 7.0
    t[-1, -1] = 9.0
    assert t[1, 1] == 9.0


def test_transpose_is_stride_swap():
    """`transpose` swaps strides without copying."""
    t = Tensor.arange(6).reshape([2, 3])
    tt = t.transpose(0, 1)
    assert tt.shape == [3, 2]
    assert tt.strides == [1, 3]
    assert not tt.is_contiguous()
    assert tt.shares_storage_with(t)
    assert tt.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]


def test_permute():
    """`permute` reorders shape and strides."""
    t = Tensor.zeros([2, 3, 4])
    p = t.permute([2, 0, 1])
    assert p.shape == [4, 2, 3]
    assert p.strides == [1, 12, 4]


def test_squeeze_unsqueeze():
    """`squeeze` drops and `unsqueeze` inserts a size-1 dimension."""
    t = Tensor.zeros([2, 1, 3])
    assert t.squeeze(1).shape == [2, 3]
    assert t.unsqueeze(0).shape == [1, 2, 1, 3]


def test_contiguous_copies():
    """`contiguous` on a strided view copies into fresh row-major storage."""
    t = Tensor.arange(6).reshape([2, 3]).transpose(0, 1)
    c = t.contiguous()
    assert c.is_contiguous()
    assert not c.shares_storage_with(t)
    assert c.tolist() == t.tolist()


def test_storage_outlives_original():
    """A view keeps its storage alive after the original tensor is deleted."""
    t = Tensor.arange(4)
    v = t.narrow(0, 1, 2)
    sid = t.storage_id
    del t
    assert v.storage_id == sid
    assert v.tolist() == [1.0, 2.0]


def test_errors():
    """Unknown dtypes, ragged data, mismatched shapes and out-of-range indices raise."""
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], dtype="nope")
    with pytest.raises(ValueError):
        Tensor([[1, 2], [3]])
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], shape=[2, 2])
    with pytest.raises(IndexError):
        Tensor.arange(3)[10]


def test_repr_and_len():
    """`repr` shows the dtype and `len` is the first dimension's size."""
    t = Tensor.arange(6).reshape([2, 3])
    assert "dtype=f32" in repr(t)
    assert len(t) == 2


def test_every_rust_factory_is_exposed():
    """Every Rust tensor factory is exposed on `Tensor` and on `lumen`."""
    for name in ["empty", "zeros", "ones", "full", "arange"]:
        assert hasattr(Tensor, name), name
        assert callable(getattr(lumen, name)), name


def test_empty_has_the_requested_shape_dtype_and_device():
    """`empty` honours shape, dtype and device, defaults to float32, and can be filled."""
    t = lumen.empty([2, 3], dtype=lumen.int32)
    assert t.shape == [2, 3]
    assert t.dtype == "int32"
    assert t.device == "cpu:0"
    assert lumen.empty([4]).dtype == "float32"
    assert t.fill_(7).tolist() == [[7, 7, 7], [7, 7, 7]]
    e = lumen.empty([3])
    for i in range(3):
        e[i] = float(i)
    assert e.tolist() == [0.0, 1.0, 2.0]


def test_ones_is_float32_by_default_like_torch():
    """`ones` defaults to float32 and honours an explicit dtype."""
    assert lumen.ones([2]).dtype == "float32"
    assert Tensor.ones([2]).tolist() == [1.0, 1.0]
    assert lumen.ones([2], dtype=lumen.int64).dtype == "int64"
    assert lumen.ones([2], dtype=lumen.bool).tolist() == [True, True]


def test_to_converts_dtypes_and_moves():
    """``to(device=None, dtype=None)``: a conversion as
    ``cast`` converts, on the tensor's device; the same
    storage when nothing changes."""
    x = lumen.tensor([1.0, 2.5, -3.7, 1e6])
    b = x.to(dtype=lumen.bfloat16)
    assert b.dtype == "bfloat16" and b.to(dtype="float32").tolist() == [1.0, 2.5, -3.703125, 999424.0]
    assert x.to(dtype="int32").tolist() == [1, 2, -3, 1000000]
    assert x.to(dtype="float32").shares_storage_with(x) and x.to("cpu").shares_storage_with(x)
    meta = x.to("meta", "float16")
    assert (meta.device, meta.dtype) == ("meta:0", "float16")
    with pytest.raises(ValueError, match="device"):
        x.to("float16")
    for device in ("mps", "cuda"):
        try:
            m = x.to(device, lumen.float16)
        except RuntimeError:
            continue
        assert (m.device, m.dtype) == (f"{device}:0", "float16")
        assert m.to("cpu", "float32").tolist() == [1.0, 2.5, -3.69921875, float("inf")]


def test_item():
    """``item``: a one-element tensor's element as a Python number (any
    shape of one element), else an error."""
    assert lumen.tensor([2.5]).item() == 2.5 and isinstance(lumen.tensor([2.5]).item(), float)
    assert lumen.tensor([[7]]).item() == 7 and isinstance(lumen.tensor([[7]]).item(), int)
    assert lumen.tensor([True]).item() is True
    assert lumen.tensor(1.5, dtype="bfloat16").item() == 1.5
    with pytest.raises(RuntimeError, match="2 elements cannot be converted to Scalar"):
        lumen.tensor([1.0, 2.0]).item()


@pytest.mark.parametrize(
    "device", [pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_item_cpu_and_to_numpy_read_device_results(device, triton_on_cuda):
    """A compiled device function's results read on the host: ``item``,
    ``cpu`` (one copy to the host) and ``to_numpy`` (that copy, through
    DLPack), each the result's values."""
    import lumen.functional as F

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    try:
        y, total = lumen.compile(lambda x: (x * 2.0, F.sum(x)), device=device)(lumen.from_numpy(a))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert y.device == f"{device}:0" and total.item() == 66.0
    host = y.cpu()
    assert host.device == "cpu:0" and host.tolist() == (a * 2).tolist()
    np.testing.assert_array_equal(lumen.to_numpy(y), a * 2)


def test_to_numpy_copies():
    """``to_numpy``: an array of the tensor's dtype, shape and values (a
    strided view's, in its logical order), its own memory; a host tensor's
    ``cpu`` is itself; numpy has no bfloat16."""
    for dtype in ["float32", "float16", "float64", "int8", "int64", "uint16", "uint64", "bool"]:
        data = [[True, False], [False, True]] if dtype == "bool" else [[1, 2], [3, 4]]
        t = lumen.tensor(data, dtype=dtype)
        a = lumen.to_numpy(t)
        assert a.dtype == np.dtype(dtype) and a.tolist() == data
        a[0, 0] = a[1, 1]
        assert t.tolist() == data
    t = lumen.tensor([[1, 2], [3, 4]])
    assert t.cpu().storage_id == t.storage_id
    assert lumen.to_numpy(t.transpose(0, 1)).tolist() == [[1, 3], [2, 4]]
    assert lumen.to_numpy(lumen.tensor(2.5)).shape == ()
    with pytest.raises(TypeError, match="bfloat16"):
        lumen.to_numpy(lumen.tensor([1.0], dtype="bfloat16"))


ALL_DTYPES = [
    "float32", "float64", "float16", "bfloat16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "bool", "int2", "int4", "uint2", "uint4", "float8_e4m3fn", "float8_e5m2", "float8_e4m3b11fnuz",
]  # fmt: skip


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_every_dtype_reads_writes_and_copies_its_elements(dtype):
    """Each dtype's elements read and write through ``t[i, j]``, ``get``
    and ``set`` (negative indices count from the end) as Python numbers of
    its kind, and a transposed view's ``contiguous`` copy holds them in
    the new order."""
    if dtype == "bool":
        data, value, kind = [[True, False], [False, False]], True, bool
    elif dtype.startswith(("int", "uint")):
        data, value, kind = [[0, 1], [1, 0]], 1, int
    else:
        data, value, kind = [[0.5, 1.0], [-1.0, 0.0]], 1.5, float
    t = Tensor(data, dtype=dtype)
    assert t.ndim == 2
    assert t[0, 1] == t.get([-2, -1]) == data[0][1] and type(t[0, 1]) is kind
    t.set([1, -1], value)
    assert t[1, 1] == value
    t[1, 0] = value
    assert t.get([1, 0]) == value
    expected = [[data[0][0], value], [data[0][1], value]]
    copy = t.transpose(0, 1).contiguous()
    assert copy.is_contiguous() and copy.storage_id != t.storage_id and copy.tolist() == expected


def test_bad_indices_and_elements_raise():
    """Elements that are not numbers, a key that is neither an int nor a
    tuple of ints, and a wrong number of indices raise."""
    with pytest.raises(TypeError, match="cannot infer dtype of element 'a'"):
        Tensor(["a"])
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="indices must be an int"):
        t[1.5]
    with pytest.raises(IndexError, match="expected 2 indices, got 1"):
        t.get([1])
    with pytest.raises(IndexError, match="expected 2 indices, got 3"):
        t.set([0, 0, 0], 1.0)


def test_full_of_a_traced_scalar_broadcasts_it():
    """Inside a trace, ``lumen.full`` of a 0-d traced value broadcasts it,
    cast to ``dtype`` when given."""
    import lumen.functional as F

    x = lumen.tensor([1.0, 2.0, 3.5])
    same, cast = lumen.compile(lambda x: (lumen.full([2, 2], F.sum(x)), lumen.full([3], F.sum(x), dtype="int32")))(x)
    assert (same.dtype, same.shape, same.tolist()) == ("float32", [2, 2], [[6.5, 6.5], [6.5, 6.5]])
    assert (cast.dtype, cast.tolist()) == ("int32", [6, 6, 6])


def test_numpy_array_protocol_casts_to_the_dtype_asked():
    """``np.asarray(t, dtype)`` reads the tensor and casts it to ``dtype``."""
    a = np.asarray(lumen.tensor([1, 2]), dtype=np.float64)
    assert a.dtype == np.float64 and a.tolist() == [1.0, 2.0]
    assert np.asarray(lumen.tensor([1, 2])).dtype == np.int64


@pytest.mark.parametrize("dtype", ["int2", "int4", "uint2", "uint4", "float8_e4m3fn", "float8_e5m2"])
def test_numpy_round_trips_ml_dtypes(dtype):
    """Narrow dtypes numpy reads through ``ml_dtypes`` convert element by
    element, both ways."""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    data = [[1, 0], [1, 1]] if dtype.startswith("u") else [[1, -1], [0, 1]]
    t = lumen.tensor(data, dtype=dtype)
    a = lumen.to_numpy(t)
    assert a.dtype == np.dtype(getattr(ml_dtypes, dtype)) and a.astype(np.float32).tolist() == data
    back = lumen.from_numpy(a)
    assert back.dtype == dtype and back.tolist() == t.tolist()
