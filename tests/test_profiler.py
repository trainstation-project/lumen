"""lumen.profiler, checked against torch.profiler's API and outputs."""

import json

import pytest

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile, record_function


def _names(prof, kind=None):
    return [e["name"] for e in prof.events() if kind is None or e["kind"] == kind]


def test_records_ops_and_user_ranges_nested():
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with record_function("outer"):
            lumen.zeros([2, 3])
    events = {e["name"]: e for e in prof.events()}
    assert events["outer"]["kind"] == "user_range"
    assert events["lumen::zeros"]["kind"] == "op"
    assert events["lumen::zeros"]["parent"] == events["outer"]["id"]
    assert events["lumen::empty"]["parent"] == events["lumen::zeros"]["id"]
    assert events["outer"]["duration_us"] >= events["lumen::zeros"]["duration_us"]


def test_record_function_works_as_a_decorator():
    @record_function("decorated")
    def work():
        return lumen.ones([4])

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        assert work().tolist() == [1.0] * 4
    events = {e["name"]: e for e in prof.events()}
    assert events["lumen::ones"]["parent"] == events["decorated"]["id"]


def test_record_function_is_free_without_a_profiler():
    with record_function("nobody is listening"):
        lumen.zeros([1])


def test_key_averages_rows_and_table():
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        for _ in range(3):
            lumen.zeros([256])
    rows = {r.key: r for r in prof.key_averages()}
    zeros = rows["lumen::zeros"]
    assert zeros.count == 3
    assert zeros.cpu_time_total >= zeros.self_cpu_time_total >= 0
    # Each result is freed after the op returns, outside any op, so zeros is
    # charged only the allocations (as in PyTorch).
    assert zeros.cpu_memory_usage == 3 * 1024
    assert rows["lumen::empty"].self_cpu_memory_usage == 3 * 1024
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
    for column in ["Name", "Self CPU %", "CPU total", "CPU Mem", "Self CPU Mem", "# of Calls"]:
        assert column in table
    assert "lumen::zeros" in table
    assert "Self CPU time total:" in table


def test_table_rejects_unknown_sort_keys():
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        lumen.zeros([1])
    with pytest.raises(ValueError):
        prof.key_averages().table(sort_by="flops")


def test_memory_events_carry_pytorchs_fields():
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        t = lumen.zeros([10], dtype=lumen.float64)
        del t
    memory = [e for e in prof.events() if e["kind"] == "memory"]
    assert [m["bytes"] for m in memory] == [80, -80]
    assert memory[0]["addr"] == memory[1]["addr"]
    assert all(m["device"] == "cpu" and m["name"] == "[memory]" for m in memory)


def test_record_shapes():
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        lumen.zeros([2, 5])
    zeros = next(e for e in prof.events() if e["name"] == "lumen::zeros")
    assert zeros["inputs"] == [] and zeros["outputs"] == [("float32", [2, 5])]


def test_plans_record_their_steps_types():
    f = lumen.compile(lambda x, y: F.sum(x * y, -1))
    x, y = lumen.ones([2, 3], dtype="int32"), lumen.ones([2, 3], dtype="int32")
    f(x, y)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        f(x, y)
    by_name = {e["name"]: e for e in prof.events()}
    assert "lumen::plan" not in by_name  # each step its own range
    assert by_name["mul"]["inputs"] == [("int32", [2, 3])] * 2 and by_name["mul"]["outputs"] == [("int32", [2, 3])]
    assert by_name["reduce_sum"]["outputs"] == [("int32", [2])]


def test_chrome_trace_is_valid_json_in_pytorchs_layout(tmp_path):
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        with record_function('name with "quotes"'):
            lumen.arange(8)
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    ops = [e for e in events if e.get("cat") == "cpu_op"]
    assert any(e["name"] == "lumen::arange" and e["ph"] == "X" for e in ops)
    assert any(e.get("cat") == "user_annotation" and e["name"] == 'name with "quotes"' for e in events)
    memory = [e for e in events if e["name"] == "[memory]"]
    assert memory and set(memory[0]["args"]) >= {
        "Device Type",
        "Device Id",
        "Addr",
        "Bytes",
        "Total Allocated",
        "Total Reserved",
    }


def test_only_one_profiler_at_a_time():
    with profile(activities=[ProfilerActivity.CPU]):
        with pytest.raises(RuntimeError, match="already running"):
            with profile(activities=[ProfilerActivity.CPU]):
                pass


def test_results_need_a_finished_session():
    prof = profile(activities=[ProfilerActivity.CPU])
    with pytest.raises(RuntimeError, match="not finished"):
        prof.key_averages()


def test_supported_activities_always_include_cpu():
    assert ProfilerActivity.CPU in lumen.profiler.supported_activities()


def test_unknown_activity_names_are_rejected():
    with pytest.raises(ValueError):
        profile(activities=["tpu"])


# ---------------- devices ----------------

MPS = pytest.param(ProfilerActivity.MPS, marks=pytest.mark.mps)
CUDA = pytest.param(ProfilerActivity.CUDA, marks=pytest.mark.cuda)


@pytest.mark.parametrize("activity", [MPS, CUDA])
def test_device_fills_are_timed_on_the_gpu(activity):
    if activity not in lumen.profiler.supported_activities():
        pytest.skip(f"{activity.value} not available in this build")
    with profile(activities=[ProfilerActivity.CPU, activity], profile_memory=True) as prof:
        lumen.zeros([1 << 16], device=activity.value)
    gpu = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    # MPS fills with its Fill compute kernel, as PyTorch does; CUDA with the
    # CuTe DSL kernel, whose name the DSL generates.
    assert len(gpu) == 1
    if activity == ProfilerActivity.MPS:
        assert gpu[0]["name"] == "Fill"
    assert gpu[0]["device"].startswith(activity.value)
    zeros = next(e for e in prof.events() if e["name"] == "lumen::zeros")
    assert gpu[0]["parent"] == zeros["id"]
    name = activity.name  # "MPS" / "CUDA"
    table = prof.key_averages().table(sort_by="self_device_time_total")
    assert f"Self {name}" in table and f"{name} Mem" in table
    trace = json.loads(_trace(prof))
    assert any(e.get("cat") == "kernel" for e in trace["traceEvents"])
    assert any(e.get("cat") == "ac2g" for e in trace["traceEvents"])


@pytest.mark.cuda
def test_cuda_copies_are_timed_on_the_gpu():
    if ProfilerActivity.CUDA not in lumen.profiler.supported_activities():
        pytest.skip("cuda not available in this build")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t = lumen.tensor([1.0, 2.0], device="cuda")
        assert t.tolist() == [1.0, 2.0]
    assert {"Memcpy HtoD", "Memcpy DtoH"} <= set(_names(prof, "gpu"))


def _trace(prof):
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            return f.read()
    finally:
        os.remove(path)


@pytest.mark.mps
def test_kernels_record_their_steps_types():
    try:
        x = lumen.ones([4, 8], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda x: F.exp(x * 2.0))
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        f(x)
    kernels = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert kernels and all(k["outputs"] == [("float32", [4, 8])] for k in kernels)
    assert all(k["inputs"] == [("float32", [4, 8])] for k in kernels)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_steps_record_their_accumulation_dtypes(device, tmp_path):
    """With record_shapes each plan step records its inputs', outputs' and
    accumulation dtypes (``accum``): a dot's and a sum's accum_dtype, a
    max's dtype, a fusion's reductions'; none for elementwise steps. Its
    kernels on the device carry them too, and the trace has them."""
    try:
        a = lumen.ones([4, 8], device=device).to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    b = lumen.ones([8, 16], device=device).to(dtype="bfloat16")
    f = lumen.compile(lambda a, b: (F.sum((a @ b).float(), -1), F.amax(a, -1), a * 2.0))
    f(a, b)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.MPS] if device == "mps" else [])
    with profile(activities=activities, record_shapes=True) as prof:
        f(a, b)
        if device == "mps":
            lumen.mps.synchronize()
    ops = {e["name"]: e for e in prof.events() if e["kind"] == "op"}
    # The dot (its result rounded to bfloat16: the cast to float32 not in
    # its epilogue, on MPS either).
    (dot,) = [e for n, e in ops.items() if n.startswith("dot_general")]
    assert dot["inputs"] == [("bfloat16", [4, 8]), ("bfloat16", [8, 16])]
    assert dot["accum"] == ["float32"] and dot["outputs"] == [("bfloat16", [4, 16])]
    (total,) = [e for n, e in ops.items() if n.endswith("reduce_sum")]
    assert total["accum"] == ["float32"] and total["outputs"] == [("float32", [4])]
    (peak,) = [e for n, e in ops.items() if n.endswith("reduce_max")]
    assert peak["accum"] == ["bfloat16"]
    (scaled,) = [e for n, e in ops.items() if n.endswith("mul") and "reduce" not in n]
    assert scaled["accum"] == []
    if device == "mps":
        kernels = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
        by_parent = {k["parent"]: k for k in kernels}
        assert by_parent[dot["id"]]["accum"] == ["float32"]
        path = tmp_path / "trace.json"
        prof.export_chrome_trace(str(path))
        text = path.read_text()
        assert '"Accum type":["f32"]' in text and '"Accum type":["bf16"]' in text


@pytest.mark.mps
def test_split_reduction_launches_record_their_own_types():
    """A split reduction is two launches, each profiled with what it reads
    and writes: the step's input to partials (of the accumulation dtype),
    then those partials to the output, named after its op."""
    try:
        x = lumen.ones([4, 200_000], device="mps").to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda a: F.sum(a.float(), -1))
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        f(x)
        lumen.mps.synchronize()
    first, last = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    (partials,) = first["outputs"]
    assert first["inputs"] == [("bfloat16", [4, 200_000])]
    assert partials[0] == "float32" and partials[1][0] == 4 and partials[1][1] > 1
    assert last["name"] == "reduce_sum" and last["kernel"] == "reduce_sum_rows_f32"
    assert last["inputs"] == [partials] and last["outputs"] == [("float32", [4])]
    assert first["accum"] == last["accum"] == ["float32"]


@pytest.mark.mps
def test_launches_are_named_after_what_they_compute():
    """A step's event keeps its whole label; each of its launches is named
    after the primitives it runs: a split reduction with an epilogue (the
    cast back of ``x.float().sum()``) converts and sums into partials in
    its first launch, sums those and casts in its second. One launch is
    the step's."""
    sep = lumen._C.FUSION_SEPARATOR
    try:
        x = lumen.ones([1024, 1024], device="mps").to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    up, down = "cast(bfloat16 -> float32)", "cast(float32 -> bfloat16)"
    step = sep.join([up, "reduce_sum", down])
    for f, launches in [
        (
            lambda a: F.sum(a.float()).bfloat16(),
            [
                (sep.join([up, "reduce_sum"]), ("bfloat16", [1024, 1024]), "float32"),
                (sep.join(["reduce_sum", down]), "float32", ("bfloat16", [])),
            ],
        ),
        (lambda a: F.sum(a.float(), -1).bfloat16(), [(step, ("bfloat16", [1024, 1024]), ("bfloat16", [1024]))]),
    ]:
        g = lumen.compile(f)
        g(x)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
            g(x)
            lumen.mps.synchronize()
        (op,) = [e for e in prof.events() if e["kind"] == "op" and "reduce_sum" in e["name"]]
        assert op["name"] == step
        gpu = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
        assert [k["name"] for k in gpu] == [name for name, _, _ in launches]
        assert all(k["parent"] == op["id"] for k in gpu)
        for k, (_, read, written) in zip(gpu, launches):
            (inp,), (out,) = k["inputs"], k["outputs"]
            assert inp == read if isinstance(read, tuple) else inp[0] == read, (k["name"], inp)
            assert out == written if isinstance(written, tuple) else out[0] == written, (k["name"], out)


@pytest.mark.mps
def test_kernels_queued_far_ahead_are_each_timed():
    """Kernels queued ahead of the GPU are each timed: no two share a start
    and duration, nor takes its command buffer's whole span (the untimed
    fallback's times, which every op got once Metal's 32 counter sample
    buffers were all held by command buffers in flight; now pooled). How far
    the host gets ahead depends on the machine's load: this checks the
    timing, not that the pool is exercised."""
    import collections

    import numpy as np

    import lumen.functional as F

    class Weight(lumen.nn.Module):
        w: lumen.Tensor

    model = Weight(lumen.empty([1024, 1024], device="meta"))

    def chain(m, s):
        x = m.w * s
        for _ in range(32):
            x = F.relu(x @ m.w) * 0.05
        return F.sum(x)

    f = lumen.compile(chain, device="mps")
    try:
        f(model, 1.0)
    except RuntimeError as e:
        pytest.skip(str(e))
    model.load_state_dict({"w": lumen.from_numpy(np.full((1024, 1024), 1 / 1024, np.float32))})
    # Matmuls slow enough that the host queues far ahead of the GPU.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
        for _ in range(40):  # ~40 command buffers of 32 kernels, queued without waiting
            f(model, 1.0)
        lumen.mps.synchronize()
    kernels = sorted((e for e in prof.events() if e["kind"] == "gpu"), key=lambda e: e["start_us"])
    assert len(kernels) >= 40 * 33
    times = collections.Counter((e["start_us"], e["duration_us"]) for e in kernels)
    assert max(times.values()) == 1
    # Each its own duration, not its command buffer's (32 kernels' span).
    spans = sorted(e["duration_us"] for e in kernels)
    assert spans[-1] < 10 * spans[len(spans) // 2]


@pytest.mark.mps
def test_host_waits_are_profiled():
    """The host's waits are CPU ranges (the GPU does nothing for them):
    ``lumen.mps.synchronize()`` as ``lumen::synchronize``, and a read of a
    result (``item``) as ``lumen::wait`` in its transfer to the host
    (``lumen::copy_d2h``), spanning the run's kernels."""
    import numpy as np

    f = lumen.compile(lambda x: F.sum(F.exp(x @ x)), device="mps")
    x = lumen.from_numpy(np.ones((256, 256), np.float32) / 256)
    try:
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
    lumen.mps.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
        f(x)
        lumen.mps.synchronize()
        f(x).item()
    events = prof.events()
    by_id = {e["id"]: e for e in events}
    (sync,) = [e for e in events if e["name"] == "lumen::synchronize"]
    (wait,) = [e for e in events if e["name"] == "lumen::wait"]
    assert sync["kind"] == wait["kind"] == "op" and sync["device"] == wait["device"] == "cpu"
    copy = by_id[wait["parent"]]
    assert copy["name"] == "lumen::copy_d2h" and by_id[copy["parent"]]["name"] == "lumen::get"


@pytest.mark.mps
def test_device_transfers_are_profiled():
    """Transfers between the host and MPS are ranges of their own: copying a
    host input in, ``lumen::copy_h2d`` on the CPU (into staging memory, the
    copy queued) and its ``copy_h2d`` kernel on the GPU; reading a result
    back, ``lumen::copy_d2h`` on the CPU (a memcpy from shared memory: no
    kernel), its ``lumen::wait`` inside it. Each the moved type in and out."""
    import numpy as np

    f = lumen.compile(lambda x: F.exp(x @ x), device="mps")
    x = lumen.from_numpy(np.ones((64, 64), np.float32) / 64)
    try:
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
    lumen.mps.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        f(x).tolist()
    events = prof.events()
    by_id = {e["id"]: e for e in events}

    def one(name, kind):
        (event,) = [e for e in events if e["name"] == name and e["kind"] == kind]
        return event

    h2d, d2h, wait = one("lumen::copy_h2d", "op"), one("lumen::copy_d2h", "op"), one("lumen::wait", "op")
    assert h2d["inputs"] == h2d["outputs"] == [("float32", [64, 64])]
    assert d2h["inputs"] == d2h["outputs"]
    assert by_id[one("copy_h2d", "gpu")["parent"]] is h2d
    assert by_id[wait["parent"]] is d2h
    assert not [e for e in events if e["kind"] == "gpu" and "d2h" in e["name"]]


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_record_function_inside_a_compiled_function(device, tmp_path):
    """A ``record_function`` range inside a compiled function (with or as a
    decorator) holds the steps of the ops traced in it each time the plan
    runs, nested as traced, one range a call; a fused kernel is in its main
    op's (a reduction's, not its epilogue's); with MPS, each range's
    kernels span it on the device timeline too."""

    @record_function("inner")
    def inner(x, w):
        return x @ w

    def f(x, w):
        with record_function("outer"):
            y = inner(x, w)
            z = F.exp(F.sum(y, -1))
        return z + 1.0

    try:
        g = lumen.compile(f, device=device)
        x, w = lumen.ones([16, 16]).to(device), lumen.ones([16, 16]).to(device)
        g(x, w)
    except RuntimeError as e:
        pytest.skip(str(e))
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.MPS] if device == "mps" else [])
    with profile(activities=activities) as prof:
        for _ in range(2):
            g(x, w)
        if device == "mps":
            lumen.mps.synchronize()
    events = prof.events()
    by_id = {e["id"]: e for e in events}
    parent = lambda e: by_id.get(e["parent"], {}).get("name")  # noqa: E731
    ranges = [(e["name"], parent(e)) for e in events if e["kind"] == "user_range"]
    assert ranges == [("outer", None), ("inner", "outer")] * 2, ranges
    steps = {e["name"]: parent(e) for e in events if e["kind"] == "op" and not e["name"].startswith("lumen::")}
    # On MPS the sum, exp and add one kernel, in the sum's range; on the
    # CPU each its own step, the add outside the range.
    want = (
        {"reduce_sum → exp → add": "outer"}
        if device == "mps"
        else {"reduce_sum": "outer", "exp": "outer", "add": None}
    )
    assert {k: steps.get(k, "missing") for k in ["dot_general", *want]} == {"dot_general": "inner", **want}, steps
    if device == "mps":
        path = tmp_path / "trace.json"
        prof.export_chrome_trace(str(path))
        spans = [
            e["name"] for e in json.loads(path.read_text())["traceEvents"] if e.get("cat") == "gpu_user_annotation"
        ]
        assert sorted(spans) == ["inner", "inner", "outer", "outer"], spans


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_record_function_holds_its_ops_gradients_as_backward(device):
    """The gradient ops of what a ``record_function`` range traced run in
    ``name (backward)``, nested as the forward's (each range of the
    forward's scope), after the forward's ranges: the matmul's gradient in
    ``outer (backward)`` / ``inner (backward)``."""

    def f(x, w):
        with record_function("outer"):
            with record_function("inner"):
                y = x @ w
            z = F.exp(F.sum(y, -1))
        return F.sum(z)

    try:
        g = lumen.compile(lumen.grad(f, (0, 1)), device=device)
        x, w = lumen.full([16, 16], 0.01).to(device), lumen.full([16, 16], 0.01).to(device)
        g(x, w)
    except RuntimeError as e:
        pytest.skip(str(e))
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.MPS] if device == "mps" else [])
    with profile(activities=activities) as prof:
        g(x, w)
        if device == "mps":
            lumen.mps.synchronize()
    events = prof.events()
    by_id = {e["id"]: e for e in events}

    def ranges(e):
        names = []
        while e.get("parent") in by_id:
            e = by_id[e["parent"]]
            names.append(e["name"])
        return names[::-1]

    dots = [ranges(e) for e in events if e["kind"] == "op" and e["name"] == "dot_general"]
    assert dots == [
        ["outer", "inner"],
        ["outer (backward)", "inner (backward)"],
        ["outer (backward)", "inner (backward)"],
    ], dots
    names = [e["name"] for e in events if e["kind"] == "user_range"]
    assert names[:2] == ["outer", "inner"] and set(names[2:]) == {"outer (backward)", "inner (backward)"}, names
