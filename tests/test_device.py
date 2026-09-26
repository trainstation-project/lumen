"""lumen.device, checked against torch.device semantics."""

import pickle

import pytest

import lumen


@pytest.mark.parametrize(
    "spec, type_, index",
    [
        ("cpu", "cpu", None),
        ("cpu:0", "cpu", 0),
        ("mps", "mps", None),
        ("cuda", "cuda", None),
        ("cuda:0", "cuda", 0),
        ("cuda:3", "cuda", 3),
    ],
)
def test_parse(spec, type_, index):
    d = lumen.device(spec)
    assert d.type == type_
    assert d.index == index
    assert str(d) == spec


def test_type_and_index_form():
    assert lumen.device("cuda", 1) == lumen.device("cuda:1")
    assert lumen.device("mps", 0).index == 0


def test_copy_from_device():
    d = lumen.device("cuda:2")
    assert lumen.device(d) == d


def test_repr():
    assert repr(lumen.device("cuda:1")) == "device(type='cuda', index=1)"
    assert repr(lumen.device("mps")) == "device(type='mps')"


def test_equality_and_hash():
    assert lumen.device("cuda:0") == lumen.device("cuda", 0)
    # like torch, "cuda" (current device) and "cuda:0" are distinct values
    assert lumen.device("cuda") != lumen.device("cuda:0")
    assert lumen.device("cpu") != lumen.device("mps")
    assert len({lumen.device("cuda:0"), lumen.device("cuda", 0), lumen.device("cpu")}) == 2


def test_frozen():
    with pytest.raises(AttributeError):
        lumen.device("cpu").index = 1


def test_pickle_roundtrip():
    d = lumen.device("cuda:1")
    assert pickle.loads(pickle.dumps(d)) == d


@pytest.mark.parametrize(
    "spec",
    ["gpu", "cuda:", "cuda:x", "cuda:-1", "cuda:+1", "cuda:1:2", "CUDA", "mps:1", "cpu:1", ""],
)
def test_invalid_strings(spec):
    with pytest.raises(ValueError):
        lumen.device(spec)


def test_invalid_arguments():
    with pytest.raises(ValueError, match="must not include an index"):
        lumen.device("cuda:0", 1)
    with pytest.raises(ValueError, match="negative"):
        lumen.device("cuda", -1)
    with pytest.raises(TypeError):
        lumen.device(0)
    with pytest.raises(TypeError):
        lumen.device(lumen.device("cpu"), 0)


def test_module_and_name():
    assert lumen.device is lumen._C.device
    assert lumen.device.__module__ == "lumen"
    assert lumen.device.__name__ == "device"
