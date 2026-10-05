"""Python-side tests pinning down the same storage semantics as the Rust tests."""

import numpy as np
import pytest

import lumen
from lumen import Tensor


def test_construct_and_infer_dtype():
    t = Tensor([[0.0, 1.0], [2.0, 3.0]])
    assert t.shape == [2, 2]
    assert t.dtype == "float32"
    assert t.device == "cpu"

    t = Tensor([1, 2, 3])
    assert t.dtype == "int64"

    t = Tensor([True, False])
    assert t.dtype == "bool"


def test_explicit_dtype():
    t = Tensor([1, 2, 3], dtype="float32")
    assert t.dtype == "float32"
    assert t.tolist() == [1.0, 2.0, 3.0]


def test_extended_dtypes():
    # integer widths
    for dt in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]:
        t = Tensor([-1, 2] if dt.startswith("int") else [1, 2], dtype=dt)
        assert t.dtype == dt
        assert t[1] == 2
    # half precision: values roundtrip through Python floats
    t = Tensor([1.5, 2.5], dtype="float16")
    assert t.dtype == "float16"
    assert t.tolist() == [1.5, 2.5]
    t[0] = 3.25
    assert t[0] == 3.25
    t = Tensor([1.5], dtype="bfloat16")
    assert t.dtype == "bfloat16"
    assert t[0] == 1.5


def test_zeros_full_arange():
    assert Tensor.zeros([2, 3]).tolist() == [[0.0] * 3] * 2
    assert Tensor.full([2, 2], 7).tolist() == [[7, 7], [7, 7]]
    assert Tensor.arange(5).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert Tensor.arange(3, dtype="int32").dtype == "int32"


def test_strides_row_major():
    t = Tensor.zeros([2, 3, 4])
    assert t.strides == [12, 4, 1]
    assert t.is_contiguous()


def test_reshape_shares_storage():
    t = Tensor.arange(6)
    m = t.reshape([2, 3])
    assert t.shares_storage_with(m)
    assert t.storage_id == m.storage_id
    assert m[1, 2] == 5.0


def test_mutation_visible_through_aliases():
    t = Tensor.arange(6)
    m = t.reshape([2, 3])
    m[0, 1] = 99.0
    assert t[1] == 99.0


def test_narrow_bumps_offset():
    t = Tensor.arange(10)
    v = t.narrow(0, 3, 4)
    assert v.storage_offset == 3
    assert v.tolist() == [3.0, 4.0, 5.0, 6.0]
    assert t.shares_storage_with(v)


def test_getitem_view_and_scalar():
    t = Tensor.arange(6).reshape([2, 3])
    row = t[1]
    assert isinstance(row, Tensor)
    assert row.tolist() == [3.0, 4.0, 5.0]
    assert row.shares_storage_with(t)
    assert t[1, 2] == 5.0
    # negative indexing
    assert t[-1].tolist() == [3.0, 4.0, 5.0]


def test_setitem():
    t = Tensor.zeros([2, 2])
    t[1, 1] = 5.0
    assert t.tolist() == [[0.0, 0.0], [0.0, 5.0]]
    # single-int key on a 1-d view
    row = t[1]
    row[0] = 7.0
    assert t[1, 0] == 7.0
    # negative index
    t[-1, -1] = 9.0
    assert t[1, 1] == 9.0


def test_transpose_is_stride_swap():
    t = Tensor.arange(6).reshape([2, 3])
    tt = t.transpose(0, 1)
    assert tt.shape == [3, 2]
    assert tt.strides == [1, 3]
    assert not tt.is_contiguous()
    assert tt.shares_storage_with(t)
    assert tt.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]


def test_permute():
    t = Tensor.zeros([2, 3, 4])
    p = t.permute([2, 0, 1])
    assert p.shape == [4, 2, 3]
    assert p.strides == [1, 12, 4]


def test_squeeze_unsqueeze():
    t = Tensor.zeros([2, 1, 3])
    assert t.squeeze(1).shape == [2, 3]
    assert t.unsqueeze(0).shape == [1, 2, 1, 3]


def test_contiguous_copies():
    t = Tensor.arange(6).reshape([2, 3]).transpose(0, 1)
    c = t.contiguous()
    assert c.is_contiguous()
    assert not c.shares_storage_with(t)
    assert c.tolist() == t.tolist()


def test_storage_outlives_original():
    t = Tensor.arange(4)
    v = t.narrow(0, 1, 2)
    sid = t.storage_id
    del t
    assert v.storage_id == sid
    assert v.tolist() == [1.0, 2.0]


def test_errors():
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], dtype="nope")
    with pytest.raises(ValueError):
        Tensor([[1, 2], [3]])  # ragged
    with pytest.raises(ValueError):
        Tensor([1, 2, 3], shape=[2, 2])  # numel mismatch
    with pytest.raises(IndexError):
        Tensor.arange(3)[10]


def test_repr_and_len():
    t = Tensor.arange(6).reshape([2, 3])
    assert "dtype=f32" in repr(t)
    assert len(t) == 2


def test_every_rust_factory_is_exposed():
    for name in ["empty", "zeros", "ones", "full", "arange"]:
        assert hasattr(Tensor, name), name
        assert callable(getattr(lumen, name)), name


def test_empty_has_the_requested_shape_dtype_and_device():
    t = lumen.empty([2, 3], dtype=lumen.int32)
    assert t.shape == [2, 3]
    assert t.dtype == "int32"
    assert t.device == "cpu"
    assert lumen.empty([4]).dtype == "float32"  # torch's default dtype
    # Contents are unspecified until written; write, then read.
    assert t.fill_(7).tolist() == [[7, 7, 7], [7, 7, 7]]
    e = lumen.empty([3])
    for i in range(3):
        e[i] = float(i)
    assert e.tolist() == [0.0, 1.0, 2.0]


def test_ones_is_float32_by_default_like_torch():
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
    assert (meta.device, meta.dtype) == ("meta", "float16")
    with pytest.raises(ValueError, match="device"):
        x.to("float16")  # a device, positionally
    try:
        m = x.to("mps", lumen.float16)
    except RuntimeError:
        return
    assert (m.device, m.dtype) == ("mps", "float16")
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


@pytest.mark.mps
def test_item_cpu_and_to_numpy_read_device_results():
    """A compiled MPS function's results read on the host: ``item``,
    ``cpu`` (one copy to the host) and ``to_numpy`` (that copy, through
    DLPack), each the result's values."""
    import lumen.functional as F

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    try:
        y, total = lumen.compile(lambda x: (x * 2.0, F.sum(x)), device="mps")(lumen.from_numpy(a))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert y.device == "mps" and total.item() == 66.0
    host = y.cpu()
    assert host.device == "cpu" and host.tolist() == (a * 2).tolist()
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
