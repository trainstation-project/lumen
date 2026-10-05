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


def _mps_heavy():
    """A compiled function of a lot of MPS work (tens of ms), its weight
    placed: queued calls of it keep the GPU busy."""
    import numpy as np

    import lumen.functional as F

    class Weight(lumen.nn.Module):
        w: lumen.Tensor

    model = Weight(lumen.empty([1024, 1024], device="meta"))

    def heavy(m, x):
        for _ in range(16):
            x = F.relu(x @ m.w) * 0.03
        return F.sum(x)

    f = lumen.compile(heavy, device="mps")
    try:
        f(model, lumen.empty([1024, 1024], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    rng = np.random.default_rng(0)
    model.load_state_dict({"w": lumen.from_numpy((rng.standard_normal((1024, 1024)) / 32).astype(np.float32))})
    x = lumen.from_numpy(np.ones((1024, 1024), np.float32))
    return lambda: f(model, x)


@pytest.mark.mps
def test_mps_inputs_copy_in_without_waiting():
    """A compiled MPS function's tensor inputs are copied in stream order:
    queuing calls returns long before the GPU has run them (the host never
    waits for the previous call's kernels)."""
    import time

    call = _mps_heavy()
    call()
    lumen.mps.synchronize()
    start = time.perf_counter()
    for _ in range(4):
        call()
    queued = time.perf_counter() - start
    lumen.mps.synchronize()
    ran = time.perf_counter() - start
    assert queued < 0.5 * ran, (queued, ran)


@pytest.mark.mps
def test_mps_input_copies_run_in_order():
    """Inputs copied in without waiting still run in stream order: calls
    queued behind GPU work each read their own input (host or MPS), the
    copy into the workspace waiting in the stream for the kernels before it,
    and a host input changed after the call returns is not what it reads."""
    import numpy as np

    import lumen.functional as F

    class Total(lumen.nn.Module):
        acc: lumen.Tensor

    def add(m, x):
        m.acc.copy_(m.acc + F.sum(x))
        return m.acc

    model = Total(lumen.empty([], device="meta"))
    step = lumen.compile(add, device="mps")
    try:
        step(model, lumen.empty([4096], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    model.load_state_dict({"acc": lumen.tensor(0.0)})
    heavy = _mps_heavy()
    values = [np.full(4096, float(k), np.float32) for k in range(1, 6)]
    host = [lumen.from_numpy(v) for v in values]
    on_mps = lumen.compile(lambda x: x * 1.0, device="mps")(lumen.from_numpy(values[4]))
    heavy()  # the GPU busy: the copies below wait in the stream
    for x in host[:4]:
        step(model, x)
    host[3].fill_(1000.0)  # after its call returned: not what it read
    total = step(model, on_mps)
    assert total.item() == 4096 * (1 + 2 + 3 + 4 + 5)


@pytest.mark.mps
def test_wait_is_an_in_place_op():
    """``Tensor.wait_``: the same tensor back, its value final (the device
    work done: the read after it does not wait), profiled as an in-place op
    on it (``lumen::wait``, its type in and out)."""
    import time

    from lumen.profiler import ProfilerActivity, profile

    call = _mps_heavy()
    call()
    lumen.mps.synchronize()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        result = call()
        assert result.wait_() is result
    start = time.perf_counter()
    result.item()
    assert time.perf_counter() - start < 0.005
    (wait,) = [e for e in prof.events() if e["name"] == "lumen::wait"]
    assert wait["inputs"] == wait["outputs"] == [("float32", [])]


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_wait_primitive(device):
    """``prims.wait``: its operand's value, in place (the plan's step reads
    and writes one buffer: no memory of its own), a step of its own (the
    host waits there: no fusion across it), its cotangent passed through."""
    import numpy as np

    import lumen.functional as F
    from lumen import prims

    def f(x):
        return F.sum(prims.wait(F.exp(x) * 2.0) * 3.0)

    graph = lumen.make_graph(f)(lumen.empty([4], device="meta"))
    try:
        plan = lumen.graph.Plan(graph, device)
    except RuntimeError as e:
        pytest.skip(str(e))
    (wait,) = [s for s in plan.steps() if s["primitive"] == "wait"]
    assert wait["inputs"][0][0] == wait["output"][0]
    if device == "mps":
        assert [s["label"] for s in plan.steps()] == ["exp → mul", "wait", "mul → reduce_sum"]
    x = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    compiled = lumen.compile(f, device=device)
    assert compiled(lumen.from_numpy(x)).item() == pytest.approx(float(np.sum(np.exp(x) * 6)), rel=1e-6)
    grad = lumen.compile(lumen.grad(f), device=device)(lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(grad), 6 * np.exp(x), rtol=1e-6)
