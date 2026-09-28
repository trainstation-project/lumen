"""Tensors on devices: the device= argument and Tensor.to()."""

import pytest

import lumen


# Device cases carry a marker so each `make test-*` target runs only its own
# (`-m mps`, `-m cuda`; `test-cpu` runs `-m "not mps and not cuda"`).
MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)


def _require(device):
    """Skip unless tensors can be created on `device` in this build."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))


def test_default_device_is_cpu():
    assert lumen.zeros([2]).device == "cpu"
    assert lumen.tensor([1, 2]).device == "cpu"


@pytest.mark.parametrize("device", ["cpu", lumen.device("cpu"), None])
def test_cpu_device_forms(device):
    t = lumen.arange(3, device=device)
    assert t.device == "cpu"
    assert t.tolist() == [0.0, 1.0, 2.0]


def test_to_same_device_shares_storage():
    t = lumen.arange(4)
    assert t.to("cpu").shares_storage_with(t)


def test_invalid_device_arguments():
    with pytest.raises(ValueError):
        lumen.zeros([1], device="gpu")
    with pytest.raises(TypeError):
        lumen.zeros([1], device=0)


def test_unavailable_cuda_raises_runtime_error():
    try:
        t = lumen.zeros([1], device="cuda:0")
    except RuntimeError as e:
        assert "not available" in str(e)
    else:  # a machine with CUDA
        assert t.device == "cuda:0"


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_tensor_on_device(device):
    _require(device)
    t = lumen.tensor([[1, 2, 3], [4, 5, 6]], device=device)
    assert t.device.startswith(device)
    t[0, 1] = 9
    assert t.tolist() == [[1, 9, 3], [4, 5, 6]]
    row = t[1]
    assert row.shares_storage_with(t)
    assert row.tolist() == [4, 5, 6]


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_to_and_back(device):
    _require(device)
    cpu = lumen.arange(6).reshape([2, 3]).transpose(0, 1)
    moved = cpu.to(lumen.device(device))
    assert moved.device.startswith(device)
    assert not moved.shares_storage_with(cpu)
    assert moved.tolist() == cpu.tolist()
    assert moved.contiguous().device == moved.device
    back = moved.to("cpu")
    assert back.device == "cpu"
    assert back.tolist() == cpu.tolist()


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_factories_on_device(device):
    _require(device)
    assert lumen.zeros([2], dtype=lumen.int32, device=device).tolist() == [0, 0]
    assert lumen.ones([2], device=device).tolist() == [1.0, 1.0]
    assert lumen.full([2], 7, device=device).tolist() == [7, 7]


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_fill_and_zero_in_place(device):
    if device != "cpu":
        _require(device)
    t = lumen.arange(6, device=device).reshape([2, 3])
    assert t.fill_(1.5) is t  # returns the tensor itself, like torch
    assert t.tolist() == [[1.5] * 3] * 2
    t[1].zero_()  # a view: writes through to t
    assert t.tolist() == [[1.5] * 3, [0.0] * 3]
    t.transpose(0, 1)[0].fill_(-1)  # strided view (column 0)
    assert t.tolist() == [[-1.0, 1.5, 1.5], [-1.0, 0.0, 0.0]]
    with pytest.raises(TypeError):
        t.fill_("x")


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_fill_strided_higher_rank(device):
    # The strided kernels decode the inner dims per thread; exercise 3-D and
    # 4-D views (permuted, so non-contiguous on every axis but one) rather
    # than only the 2-D transpose above.
    _require(device)
    for shape, perm, value in [
        ([2, 3, 4], [2, 0, 1], -1.5),
        ([2, 3, 4, 5], [3, 1, 0, 2], 7.0),
    ]:
        n = 1
        for s in shape:
            n *= s
        t = lumen.arange(n, dtype=lumen.float32, device=device).reshape(shape)
        view = t.permute(perm)
        assert not view.is_contiguous()
        view.fill_(value)
        flat = view.to("cpu").tolist()

        def flatten(xs):
            for x in xs:
                if isinstance(x, list):
                    yield from flatten(x)
                else:
                    yield x

        assert list(flatten(flat)) == [value] * n


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_empty_and_ones_on_device(device):
    _require(device)
    e = lumen.empty([2, 2], dtype=lumen.int64, device=device)
    assert e.device.startswith(device)
    assert e.zero_().tolist() == [[0, 0], [0, 0]]
    o = lumen.ones([3], device=device)
    assert (o.dtype, o.tolist()) == ("float32", [1.0, 1.0, 1.0])
