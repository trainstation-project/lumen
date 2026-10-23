import numpy as np
import pytest

import lumen

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)
TPU = pytest.param("tpu", marks=pytest.mark.tpu)


def _require(device):
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))


def test_default_device_is_cpu():
    assert lumen.zeros([2]).device == "cpu:0"
    assert lumen.tensor([1, 2]).device == "cpu:0"


@pytest.mark.parametrize("device", ["cpu", lumen.device("cpu"), None])
def test_cpu_device_forms(device):
    t = lumen.arange(3, device=device)
    assert t.device == "cpu:0"
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
    else:
        assert t.device == "cuda:0"


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_tensor_on_device(device):
    _require(device)
    t = lumen.tensor([[1, 2, 3], [4, 5, 6]], device=device)
    assert t.device.startswith(device)
    t[0, 1] = 9
    assert t.tolist() == [[1, 9, 3], [4, 5, 6]]
    row = t[1]
    assert row.shares_storage_with(t)
    assert row.tolist() == [4, 5, 6]


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_to_and_back(device):
    _require(device)
    cpu = lumen.arange(6).reshape([2, 3]).transpose(0, 1)
    moved = cpu.to(lumen.device(device))
    assert moved.device.startswith(device)
    assert not moved.shares_storage_with(cpu)
    assert moved.tolist() == cpu.tolist()
    assert moved.contiguous().device == moved.device
    back = moved.to("cpu")
    assert back.device == "cpu:0"
    assert back.tolist() == cpu.tolist()


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_factories_on_device(device):
    _require(device)
    assert lumen.zeros([2], dtype=lumen.int32, device=device).tolist() == [0, 0]
    assert lumen.ones([2], device=device).tolist() == [1.0, 1.0]
    assert lumen.full([2], 7, device=device).tolist() == [7, 7]


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_fill_and_zero_in_place(device):
    if device != "cpu":
        _require(device)
    t = lumen.arange(6, device=device).reshape([2, 3])
    assert t.fill_(1.5) is t
    assert t.tolist() == [[1.5] * 3] * 2
    t[1].zero_()
    assert t.tolist() == [[1.5] * 3, [0.0] * 3]
    t.transpose(0, 1)[0].fill_(-1)
    assert t.tolist() == [[-1.0, 1.5, 1.5], [-1.0, 0.0, 0.0]]
    with pytest.raises(TypeError):
        t.fill_("x")


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_fill_strided_higher_rank(device):
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
    t = lumen.zeros([4, 5, 6], device=device)
    t.narrow(2, 1, 3).permute([2, 0, 1]).fill_(2.0)
    rows = t.to("cpu").tolist()
    assert all(rows[i][j][k] == (2.0 if 1 <= k < 4 else 0.0) for i in range(4) for j in range(5) for k in range(6))


@pytest.mark.cuda
def test_cuda_fill_launches_on_the_tensors_device():
    _require("cuda")
    try:
        lumen.zeros([1], device="cuda:1")
    except RuntimeError:
        pytest.skip("needs two CUDA devices")
    for device, other in (("cuda:0", "cuda:1"), ("cuda:1", "cuda:0")):
        t = lumen.zeros([1 << 16], device=device)
        lumen.cuda.synchronize(other)
        t.fill_(5.0)
        assert t.tolist() == [5.0] * (1 << 16)


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_empty_and_ones_on_device(device):
    _require(device)
    e = lumen.empty([2, 2], dtype=lumen.int64, device=device)
    assert e.device.startswith(device)
    assert e.zero_().tolist() == [[0, 0], [0, 0]]
    o = lumen.ones([3], device=device)
    assert (o.dtype, o.tolist()) == ("float32", [1.0, 1.0, 1.0])


@pytest.mark.parametrize("shape", [[1], [128], [129], [10, 100]])
@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_copies_round_trip_without_touching_their_neighbours(device, shape):
    _require(device)
    rng = np.random.default_rng(0)
    hosts = [rng.standard_normal(shape, dtype=np.float32) for _ in range(3)]
    placed = [lumen.from_numpy(h).to(device) for h in hosts]
    for host, t in zip(hosts, placed):
        assert t.device == f"{device}:0"
        np.testing.assert_array_equal(lumen.to_numpy(t.to("cpu")), host)
    bf16 = lumen.from_numpy(hosts[0]).to(dtype="bfloat16")
    on_host = lambda t: lumen.to_numpy(t.to("cpu").to(dtype="float32"))  # noqa: E731
    np.testing.assert_array_equal(on_host(bf16.to(device)), on_host(bf16))
