"""Device streams: lumen.mps.synchronize and lumen.cuda.synchronize."""

import pytest

import lumen


def test_stream_modules_mirror_torch():
    assert lumen.mps is lumen.stream.mps
    assert lumen.cuda is lumen.stream.cuda


def test_cuda_synchronize_rejects_other_devices():
    with pytest.raises(ValueError, match="cuda device"):
        lumen.cuda.synchronize("cpu")


def test_cuda_synchronize_without_the_device_raises():
    try:
        lumen.zeros([1], device="cuda:7")
    except RuntimeError:
        with pytest.raises(RuntimeError):
            lumen.cuda.synchronize(7)
    else:
        pytest.skip("cuda:7 exists")


@pytest.mark.mps
def test_mps_synchronize_waits_for_queued_work():
    try:
        t = lumen.zeros([1 << 20], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    for i in range(100):
        t.fill_(float(i))  # queued, not waited on
    lumen.mps.synchronize()
    assert t[0] == 99.0 and t[(1 << 20) - 1] == 99.0


@pytest.mark.cuda
@pytest.mark.parametrize("device", [None, 0, "cuda", "cuda:0", lumen.device("cuda", 0)])
def test_cuda_synchronize_forms(device):
    try:
        t = lumen.zeros([1 << 20], device="cuda")
    except RuntimeError as e:
        pytest.skip(str(e))
    t.fill_(3.0)
    lumen.cuda.synchronize(device)
    assert t[0] == 3.0
