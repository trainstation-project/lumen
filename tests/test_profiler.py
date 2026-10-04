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
    assert by_name["lumen::plan"]["inputs"] == [("int32", [2, 3])] * 2
    assert by_name["lumen::plan"]["outputs"] == [("int32", [2])]
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
    gpu = [e for e in prof.events() if e["kind"] == "gpu"]
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
    kernels = [e for e in prof.events() if e["kind"] == "gpu"]
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
    # The dot (on MPS with its epilogue, the cast to float32, fused in).
    (dot,) = [e for n, e in ops.items() if n.startswith("dot_general")]
    assert dot["inputs"] == [("bfloat16", [4, 8]), ("bfloat16", [8, 16])]
    out = "float32" if device == "mps" else "bfloat16"
    assert dot["accum"] == ["float32"] and dot["outputs"] == [(out, [4, 16])]
    (total,) = [e for n, e in ops.items() if n.endswith("reduce_sum")]
    assert total["accum"] == ["float32"] and total["outputs"] == [("float32", [4])]
    (peak,) = [e for n, e in ops.items() if n.endswith("reduce_max")]
    assert peak["accum"] == ["bfloat16"]
    (scaled,) = [e for n, e in ops.items() if n.endswith("mul") and "reduce" not in n]
    assert scaled["accum"] == []
    if device == "mps":
        kernels = [e for e in prof.events() if e["kind"] == "gpu"]
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
    first, last = [e for e in prof.events() if e["kind"] == "gpu"]
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
    step = sep.join(["convert_element_type", "reduce_sum", "convert_element_type"])
    for f, launches in [
        (
            lambda a: F.sum(a.float()).bfloat16(),
            [
                (sep.join(["convert_element_type", "reduce_sum"]), ("bfloat16", [1024, 1024]), "float32"),
                (sep.join(["reduce_sum", "convert_element_type"]), "float32", ("bfloat16", [])),
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
        gpu = [e for e in prof.events() if e["kind"] == "gpu"]
        assert [k["name"] for k in gpu] == [name for name, _, _ in launches]
        assert all(k["parent"] == op["id"] for k in gpu)
        for k, (_, read, written) in zip(gpu, launches):
            (inp,), (out,) = k["inputs"], k["outputs"]
            assert inp == read if isinstance(read, tuple) else inp[0] == read, (k["name"], inp)
            assert out == written if isinstance(written, tuple) else out[0] == written, (k["name"], out)
