"""Python-side tests pinning down the same storage semantics as the Rust tests."""

import pytest

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
