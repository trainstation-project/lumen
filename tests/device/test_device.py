"""Tests of `lumen.device` parsing, equality, pickling and the per-device availability modules."""

import pickle

import pytest

import lumen
from lumen.device import _C


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
    """Device strings parse into their type and optional index and print back unchanged."""
    d = lumen.device(spec)
    assert d.type == type_
    assert d.index == index
    assert str(d) == spec


def test_type_and_index_form():
    """`device(type, index)` equals the `type:index` string form."""
    assert lumen.device("cuda", 1) == lumen.device("cuda:1")
    assert lumen.device("mps", 0).index == 0


def test_copy_from_device():
    """A device built from another device equals it."""
    d = lumen.device("cuda:2")
    assert lumen.device(d) == d


def test_repr():
    """The repr shows the type and, when set, the index."""
    assert repr(lumen.device("cuda:1")) == "device(type='cuda', index=1)"
    assert repr(lumen.device("mps")) == "device(type='mps')"


def test_equality_and_hash():
    """Devices compare and hash by type and index, so an unindexed device differs from index 0."""
    assert lumen.device("cuda:0") == lumen.device("cuda", 0)
    assert lumen.device("cuda") != lumen.device("cuda:0")
    assert lumen.device("cpu") != lumen.device("mps")
    assert len({lumen.device("cuda:0"), lumen.device("cuda", 0), lumen.device("cpu")}) == 2


def test_frozen():
    """Device attributes are read-only."""
    with pytest.raises(AttributeError):
        lumen.device("cpu").index = 1


def test_pickle_roundtrip():
    """Devices survive pickling."""
    d = lumen.device("cuda:1")
    assert pickle.loads(pickle.dumps(d)) == d


@pytest.mark.parametrize(
    "spec",
    ["gpu", "cuda:", "cuda:x", "cuda:-1", "cuda:+1", "cuda:1:2", "CUDA", ""],
)
def test_invalid_strings(spec):
    """Malformed device strings raise `ValueError`."""
    with pytest.raises(ValueError):
        lumen.device(spec)


@pytest.mark.parametrize("spec", ["cpu:1", "mps:1", "meta:1"])
def test_every_device_is_numbered(spec):
    """Any device type takes an index, and placing a tensor on a missing one raises `RuntimeError`."""
    assert lumen.device(spec).index == 1
    with pytest.raises(RuntimeError, match="device 1 is not available|MPS is not available"):
        lumen.zeros([2], device=spec)


def test_invalid_arguments():
    """Contradictory or mistyped constructor arguments raise `ValueError` or `TypeError`."""
    with pytest.raises(ValueError, match="must not include an index"):
        lumen.device("cuda:0", 1)
    with pytest.raises(ValueError, match="negative"):
        lumen.device("cuda", -1)
    with pytest.raises(TypeError):
        lumen.device(0)
    with pytest.raises(TypeError):
        lumen.device(lumen.device("cpu"), 0)


def test_module_and_name():
    """`lumen.device` is the native class exported under the `lumen` module."""
    assert lumen.device is _C.device
    assert lumen.device.__module__ == "lumen"
    assert lumen.device.__name__ == "device"


def test_is_available():
    """Each device type's module (``lumen.cuda``, PyTorch's ``torch.cuda``):
    available where it can hold tensors in this build."""
    import lumen.cuda

    assert lumen.cpu.device_count() == lumen.meta.device_count() == 1
    with pytest.raises(ValueError, match="not a cuda device"):
        lumen.cuda.synchronize("cpu")
    for module in [lumen.cpu, lumen.meta, lumen.cuda, lumen.mps]:
        try:
            lumen.zeros([1], device=module.__name__.split(".")[-1])
            placed = True
        except RuntimeError:
            placed = False
        assert module.is_available() == placed == (module.device_count() > 0)
        if placed:
            module.synchronize()
