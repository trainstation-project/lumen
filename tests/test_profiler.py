"""lumen.profiler, checked against torch.profiler's API and outputs."""

import json

import pytest

import lumen
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
    assert zeros["shapes"] == [[2, 5]]


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
    assert any(
        e.get("cat") == "user_annotation" and e["name"] == 'name with "quotes"' for e in events
    )
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

# CUDA fills are CuTe DSL kernels, which the CUPTI setup does not record
# (it records copies and memsets), so only MPS fills are timed here.
@pytest.mark.mps
def test_device_fills_are_timed_on_the_gpu():
    activity = ProfilerActivity.MPS
    if activity not in lumen.profiler.supported_activities():
        pytest.skip(f"{activity.value} not available in this build")
    with profile(activities=[ProfilerActivity.CPU, activity], profile_memory=True) as prof:
        lumen.zeros([1 << 16], device=activity.value)
    gpu = [e for e in prof.events() if e["kind"] == "gpu"]
    # MPS fills with a compute kernel, as PyTorch does.
    assert [e["name"] for e in gpu] == ["Fill"]
    assert gpu[0]["device"].startswith(activity.value)
    zeros = next(e for e in prof.events() if e["name"] == "lumen::zeros")
    assert gpu[0]["parent"] == zeros["id"]
    table = prof.key_averages().table(sort_by="self_device_time_total")
    assert "Self MPS" in table and "MPS Mem" in table
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
