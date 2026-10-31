"""Tests of lumen streams, events and the ``wait`` primitive."""

import pytest
from helpers import CUDA, MPS, TPU

import lumen


def test_every_device_has_a_stream():
    """``lumen.Stream(device)``: the device's stream, which every device type
    has (the CPU's and the meta device's have no work to wait for); a
    device that is not present has none."""
    for spec in ["cpu", "meta:0", "mps", "cuda"]:
        try:
            stream = lumen.Stream(spec)
        except RuntimeError:
            continue
        assert stream.device == lumen.device(lumen.device(spec).type, 0) and stream.stream_id == 0
        assert repr(stream) == f"lumen.Stream(device='{stream.device}')"
        stream.synchronize()
    assert lumen.Stream().device == lumen.device("cpu", 0)
    with pytest.raises(RuntimeError, match="not available"):
        lumen.Stream("cpu:1")


def test_the_stream_is_the_only_api():
    """``lumen.Stream`` and ``lumen.Event`` are the stream API (no
    per-device stream functions)."""
    assert lumen.stream.__all__ == ["Event", "Stream"]
    cpu = lumen.Stream("cpu")
    assert cpu == lumen.Stream("cpu:0") and hash(cpu) == hash(lumen.Stream()) and cpu != lumen.Stream("meta")
    cpu.wait_stream(lumen.Stream("cpu"))
    cpu.wait_stream(lumen.Stream("meta"))
    with pytest.raises(ValueError, match="an external stream is CUDA's"):
        lumen.Stream("cpu", stream_id=1)


@pytest.mark.parametrize(
    "device",
    ["cpu", MPS, TPU, CUDA],
)
def test_events_time_the_work_between_them(device, triton_on_cuda):
    """Two events with ``enable_timing`` around a compiled program:
    ``elapsed_time`` is the device time between them in milliseconds,
    longer for more work, and ``query`` holds once the later is done. The
    first calls compile the program and, on CUDA, capture it as a graph,
    so they run before the timing."""
    try:
        x = lumen.ones([1024, 4096], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda x: x * 2.0 + 1.0)
    for _ in range(2):
        f(x)
    lumen.Stream(device).synchronize()

    def timed(reps):
        """Device milliseconds between events around ``reps`` calls."""
        start, end = lumen.Event(device, enable_timing=True), lumen.Event(device, enable_timing=True)
        start.record()
        for _ in range(reps):
            f(x)
        end.record()
        end.synchronize()
        assert end.query() and start.query()
        return start.elapsed_time(end)

    one, many = timed(1), timed(20)
    assert 0 < one < many


def test_events_need_timing_and_a_record():
    """``elapsed_time`` needs both events recorded and timed; the meta
    device runs nothing to record an event after."""
    plain, timed = lumen.Event("cpu"), lumen.Event("cpu", enable_timing=True)
    assert plain.query() and not plain.enable_timing and timed.enable_timing
    assert repr(plain) == "lumen.Event(device='cpu:0', enable_timing=False)"
    with pytest.raises(RuntimeError, match="record both"):
        timed.elapsed_time(lumen.Event("cpu", enable_timing=True))
    plain.record()
    timed.record()
    with pytest.raises(RuntimeError, match="enable_timing=True"):
        plain.elapsed_time(timed)
    with pytest.raises(RuntimeError, match="meta device"):
        lumen.Event("meta")


def test_a_stream_of_an_absent_device_raises():
    """A stream of a device index that isn't present raises."""
    try:
        lumen.zeros([1], device="cuda:7")
    except RuntimeError:
        with pytest.raises(RuntimeError, match="not available"):
            lumen.Stream("cuda:7")
    else:
        pytest.skip("cuda:7 exists")


@pytest.mark.mps
def test_mps_synchronize_waits_for_queued_work():
    """``lumen.mps.synchronize`` returns only after every queued fill has run."""
    try:
        t = lumen.zeros([1 << 20], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    for i in range(100):
        t.fill_(float(i))
    lumen.mps.synchronize()
    assert t[0] == 99.0 and t[(1 << 20) - 1] == 99.0


@pytest.mark.cuda
@pytest.mark.parametrize("device", ["cuda", "cuda:0", lumen.device("cuda", 0)])
def test_cuda_stream_forms(device):
    """A CUDA stream can be named by type, by ``type:index`` or by a device."""
    try:
        t = lumen.zeros([1 << 20], device="cuda")
    except RuntimeError as e:
        pytest.skip(str(e))
    t.fill_(3.0)
    lumen.Stream(device).synchronize()
    assert t[0] == 3.0


def _mps_heavy():
    """A compiled function of a lot of MPS work (tens of ms), its weight
    placed: queued calls of it keep the GPU busy."""
    import numpy as np

    import lumen.functional as F

    class Weight(lumen.nn.Module):
        """A module holding one weight."""

        w: lumen.Tensor

    model = Weight(lumen.empty([1024, 1024], device="meta"))

    def heavy(m, x):
        """Sixteen ReLU matmul layers, summed."""
        for _ in range(16):
            x = F.relu(F.matmul(x, m.w, "float32", "float32")) * 0.03
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
        """A module holding a running total."""

        acc: lumen.Tensor

    def add(m, x):
        """Adds ``sum(x)`` into ``m.acc`` in place and returns it."""
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
    heavy()
    for x in host[:4]:
        step(model, x)
    host[3].fill_(1000.0)
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


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_wait_primitive(device, triton_on_cuda):
    """``prims.host_wait``: its operand's value, in place (the plan's step reads
    and writes one buffer: no memory of its own), a step of its own (the
    host waits there: no fusion across it), its cotangent passed through."""
    import numpy as np

    import lumen.functional as F
    from lumen import prims

    def f(x):
        """``sum(wait(2 exp x) * 3)``."""
        return F.sum(prims.host_wait(F.exp(x) * 2.0) * 3.0)

    graph = lumen.make_graph(f)(lumen.empty([4], device="meta"))
    try:
        plan = lumen.graph_compiler.Plan(graph, device)
    except RuntimeError as e:
        pytest.skip(str(e))
    (wait,) = [s for s in plan.steps() if s["primitive"] == "host_wait"]
    assert wait["inputs"][0][0] == wait["output"][0]
    if device == "mps":
        assert [s["label"] for s in plan.steps()] == ["exp → mul", "host_wait", "mul → reduce_sum"]
    x = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    compiled = lumen.compile(f, device=device)
    assert compiled(lumen.from_numpy(x)).item() == pytest.approx(float(np.sum(np.exp(x) * 6)), rel=1e-6)
    grad = lumen.compile(lumen.grad(f), device=device)(lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(grad), 6 * np.exp(x), rtol=1e-6)
