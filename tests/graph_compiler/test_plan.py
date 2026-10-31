"""Tests of compiled plans: steps, workspace, results, stages, devices, strided inputs, autotuning and dtypes."""

import re

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, rand

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_autotuned_loops_match_untuned(device, dtype, tmp_path, triton_on_cuda):
    """A loop the compiler autotunes computes what the untuned loop does, at
    lengths no block size divides. A Triton loop's dumped source names its
    template kernel and the ``BLOCK_SIZE`` and warps it picked, then its
    generated ``BODY``."""
    compiler = getattr(lumen.config.compiler, device)
    try:
        for n in ((1 << 16) + 7, 1 << 20, (3 << 20) + 1):
            try:
                x = lumen.from_numpy(rand(n, seed=n)).to(device).to(dtype=dtype)
            except RuntimeError as e:
                pytest.skip(str(e))
            results = []
            for autotune in (False, True):
                compiler.autotune = autotune
                y = lumen.compile(lambda x: -x * 2)(x)
                results.append(lumen.to_numpy(y.to(dtype="float32").to("cpu")))
            assert np.array_equal(results[0], results[1]), n
        if device != "cuda":
            return
        lumen.compile(lambda x: -x * 2).dump_code(str(tmp_path), x)
        source = "".join(p.read_text() for p in tmp_path.glob("*elementwise_*.py"))
        assert re.search(r"elementwise_kernel, BLOCK_SIZE=\d+ warps=\d+", source)
        assert re.search(r"def elementwise_[0-9a-f]+\(EXTRA, BLOCK, MASK, SEGMENT: tl.constexpr\)", source)
    finally:
        compiler.autotune = True


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_reads_strided_inputs_in_place(device, triton_on_cuda):
    """A transposed or narrowed input is read where it is, not copied
    first: its step's kernel is compiled again with that input's load
    prologue through its strides, once a layout. A matmul reads transposed
    and permuted operands too (on Triton through the GEMM's
    ``PROLOGUE_A``/``PROLOGUE_B``). ``contiguous``, ``copy_`` and ``fill_``
    of strided views are loops on the GPU, not host copies."""
    a, b, c = rand(64, 96), rand(96, 64, seed=1), rand(2, 3, 96, seed=2)
    x, w, z = (lumen.from_numpy(v).to(device) for v in (a, b, c))
    loop = lumen.compile(lambda u, v: F.exp(u) * 2.0 + v)
    norm = lumen.compile(lambda u: u / F.sqrt(F.mean(u * u, -1, keepdim=True) + 1e-6))
    tol = {"rtol": 1e-5, "atol": 1e-5}
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = loop(x.transpose(0, 1), w)
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(a.T) * 2 + b, **tol)
        out = loop(x.narrow(1, 8, 32), w.narrow(0, 8, 64).narrow(1, 0, 32))
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(a[:, 8:40]) * 2 + b[8:72, :32], **tol)
        out = norm(x.transpose(0, 1))
        expected = a.T / np.sqrt((a.T.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(lumen.to_numpy(out), expected, **tol)
    assert not [e for e in prof.events() if e["name"] == "lumen::contiguous"]
    dot = lumen.compile(lambda u, v: F.matmul(u, v, "float32", "float32"))
    if device == "cuda":
        import lumen.codegen.triton as triton

        before = set(triton.kernels)
    np.testing.assert_allclose(lumen.to_numpy(dot(x.transpose(0, 1), x)), a.T @ a, rtol=1e-4, atol=1e-4)
    out = lumen.to_numpy(dot(z.transpose(0, 1), w))
    np.testing.assert_allclose(out, c.transpose(1, 0, 2) @ b, rtol=1e-4, atol=1e-4)
    if device == "cuda":
        new = [triton.compiled(n, None) for n in set(triton.kernels) - before if n.startswith("prologue_")]
        assert any("//" in f.src or "*" in f.src for f in new)
    y = lumen.zeros([96, 64], device=device)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        t = x.transpose(0, 1).contiguous()
        y.transpose(0, 1).copy_(x)
        y.transpose(0, 1)[3].fill_(-2.0)
    host = {"lumen::to_vec", "lumen::copy_h2d", "lumen::copy_d2h"}
    assert not [e for e in prof.events() if e["name"] in host]
    assert t.is_contiguous() and lumen.to_numpy(t).tolist() == a.T.tolist()
    expected = a.T.copy()
    expected[:, 3] = -2.0
    assert lumen.to_numpy(y).tolist() == expected.tolist()


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_steps_replay_on_each_calls_buffers(device, triton_on_cuda):
    """An MPS plan's steps are encoded once and replayed after: each later
    call computes on its own inputs and workspace (a strided call between
    them too), as the first did."""
    f = lumen.compile(lambda x, w: F.sum(F.relu(F.matmul(x, w, "float32", "float32")), -1))
    expected = lambda a, b: np.maximum(a @ b, 0).sum(-1)  # noqa: E731
    for seed in range(3):
        a, b = rand(32, 64, seed=seed), rand(64, 48, seed=seed + 10)
        x, w = lumen.from_numpy(a).to(device), lumen.from_numpy(b).to(device)
        np.testing.assert_allclose(lumen.to_numpy(f(x, w)), expected(a, b), rtol=1e-4, atol=1e-4)
        t = lumen.from_numpy(np.ascontiguousarray(b.T)).to(device).transpose(0, 1)
        np.testing.assert_allclose(lumen.to_numpy(f(x, t)), expected(a, b), rtol=1e-4, atol=1e-4)
    a = rand(4096)
    graph = lumen.make_graph(lambda u: F.exp(u) * 2.0 + 1.0)(lumen.from_numpy(a).to(device))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=[0])
    workspaces = [lumen.Tensor.empty([plan.workspace_bytes], "uint8", device) for _ in range(2)]
    for k in range(4):
        v = a + k
        (out,) = plan.run_in(workspaces[k % 2], [lumen.from_numpy(v).to(device)])
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(v) * 2 + 1, rtol=1e-5)


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_results_on_input_device(device, triton_on_cuda):
    """A compiled program's result lands on its input's device."""
    try:
        x = lumen.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    y = lumen.compile(lambda x: F.matmul(x, x, "float32", x.dtype) + 1)(x)
    assert y.device == str(x.device)
    assert lumen.to_numpy(y).tolist() == [[8.0, 11.0], [16.0, 23.0]]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_compiled_blocks_match_cpu(device, dtype, triton_on_cuda):
    """Compiled functions on MPS (Metal kernels) and CUDA (Triton) agree with the CPU."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def block(x, wq, wk, w1):
        """Attention-like block with a residual, returning a row mean, column max and absolute sum."""
        scores = (
            F.matmul(F.matmul(x, wq, "float32", x.dtype), F.matmul(x, wk, "float32", x.dtype).t(), "float32", x.dtype)
            * 0.25
        )
        h = x + F.matmul(F.softmax(scores.float(), -1).to(dtype=x.dtype), x, "float32", x.dtype)
        return F.mean(F.relu(F.matmul(h, w1, "float32", x.dtype)), -1), F.amax(h, 0), F.sum(F.where(h > 0, h, -h))

    arrays = [rand(8, 16, seed=1), rand(16, 16, seed=2), rand(16, 16, seed=3), rand(16, 32, seed=4)]
    cpu = [lumen.from_numpy(a) for a in arrays]
    fn = lumen.compile(lambda *xs: tuple(o.float() for o in block(*(x.to(dtype) for x in xs))))
    expected = [lumen.to_numpy(o) for o in fn(*cpu)]
    actual = fn(*(t.to(device) for t in cpu))
    tol = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}[dtype]
    for e, a in zip(expected, actual):
        assert a.device == f"{device}:0"
        np.testing.assert_allclose(lumen.to_numpy(a), e, rtol=tol, atol=tol)


def test_plan():
    """A plan folds the reshape away, sizes its workspace and checks input dtypes when run."""
    graph = lumen.make_graph(lambda x: F.tanh((F.exp(x) + 1).reshape(-1)))(lumen.zeros([16, 16]))
    plan = lumen.graph_compiler.Plan(graph)
    assert "reshape" not in str(plan)
    assert plan.workspace_bytes == 1024 + 4, plan
    x = rand(16, 16)
    np.testing.assert_allclose(
        lumen.to_numpy(plan.run([lumen.from_numpy(x)])[0]), np.tanh(np.exp(x) + 1).ravel(), rtol=1e-6
    )
    with pytest.raises(ValueError, match="must be f32"):
        plan.run([lumen.zeros([16, 16], dtype="float64")])


def test_graph_and_plan_describe_themselves():
    """`Graph.nodes` and `Plan.steps` describe primitives, inputs and outputs."""
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x * 2.0), -1))(lumen.zeros([2, 3]))
    nodes = graph.nodes()
    assert [n["primitive"] for n in nodes] == ["full", "broadcast_in_dim", "mul", "exp", "reduce_sum"]
    assert nodes[-1]["text"] == "reduce_sum[axes=(1,) accum_dtype=f32]" and nodes[-1]["fusion"] is None
    assert graph.inputs() == [0] and graph.outputs() == [nodes[-1]["output"]]
    steps = lumen.graph_compiler.Plan(graph).steps()
    assert steps[-1]["output"] == ("out0", "float32", [2])
    assert steps[2]["inputs"][0] == ("in0", "float32", [2, 3])


@pytest.mark.mps
def test_mps_plan_steps_and_profiled_kernels():
    """An MPS plan of a row reduction is one fused `reduce_rows` kernel, profiled under its kernel name."""
    try:
        x = lumen.ones([64, 2048], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    from lumen.profiler import ProfilerActivity, profile

    graph = lumen.make_graph(lambda x: F.sum(F.tanh(x * 2.0 + 1.0), -1))(x)
    plan = lumen.graph_compiler.Plan(graph, "mps")
    (step,) = plan.steps()
    fusion = step["fusion"]
    assert fusion["kernel"] in fusion["source"] and "tanh" in fusion["body"] and "reduce_sum" in fusion["body"]
    assert "reduce_rows" in fusion["source"]
    plan.run([x])
    lumen.mps.synchronize()
    with profile(activities=[ProfilerActivity.MPS]) as prof:
        plan.run([x])
        lumen.mps.synchronize()
    kernels = [e["kernel"] for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert kernels == [fusion["kernel"]]
    assert plan.run([x])[0].tolist() == pytest.approx([float(np.tanh(3.0)) * 2048] * 64, rel=1e-5)


def test_compiled_init_runs_once_per_call(tmp_path):
    """An input-less compiled program traces once, and its result is a view of the workspace."""
    calls = []

    def init():
        """Counts its calls and builds a constant."""
        calls.append(1)
        return lumen.full([2, 3], 0.5) + lumen.arange(3)

    make = lumen.compile(init)
    a = make().clone()
    b = make()
    assert calls == [1]
    assert a.tolist() == b.tolist() == [[0.5, 1.5, 2.5]] * 2 and b.shares_storage_with(make())
    data = make.dump_graph(tmp_path / "init.html")
    assert data["inputs"] == [] and data["device"] == "cpu:0"


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_compiled_init_on_device(device, tmp_path, triton_on_cuda):
    """Input-less programs run on the device they are compiled for, and graph dumps report it."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    make = lumen.compile(lambda: (lumen.ones([4, 8]) * 2.0, lumen.arange(8)), device=device)
    w, i = make()
    assert (w.device, i.device) == (f"{device}:0", f"{device}:0")
    assert w.tolist() == [[2.0] * 8] * 4 and i.tolist() == list(map(float, range(8)))
    assert lumen.compile(lambda x: x + 1.0, device=device)(lumen.zeros([2])).tolist() == [1.0, 1.0]
    data = make.dump_graph(tmp_path / "init.html")
    assert data["device"] == f"{device}:0"
    if device == "mps":
        assert any(n.get("kernels") for n in data["views"]["fused"]["nodes"])
    f = lumen.compile(lambda x, w: F.matmul(x, w, "float32", x.dtype))
    meta = lumen.empty([2, 4], device="meta"), lumen.empty([4, 3], device="meta")
    data = f.dump_graph(tmp_path / "f.html", *meta, device=device)
    assert data["device"] == f"{device}:0" and data["inputs"] == ["f32[2,4]", "f32[4,3]"]
    if device == "mps":
        assert any(n.get("kernels") for n in data["views"]["fused"]["nodes"])


def test_results_are_views_of_the_workspace():
    """Each call's result is a view of the same workspace, overwritten by the next call."""
    f = lumen.compile(lambda x: x * 2.0)
    a = f(lumen.ones([3]))
    assert a.tolist() == [2.0] * 3
    b = f(lumen.full([3], 5.0))
    assert a.shares_storage_with(b) and a.tolist() == [10.0] * 3
    assert not a.clone().shares_storage_with(b)


@pytest.mark.mps
def test_mps_plans_put_copies_and_scratch_in_the_workspace():
    """MPS plans transpose dot operands as separate steps and give large reductions workspace scratch."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    dot = lambda a, b: prims.dot_general(a, b, (((0, 2), (2, 0)), ((), ())), a.dtype, a.dtype)  # noqa: E731
    graph = lumen.make_graph(dot)(lumen.zeros([3, 2, 4]), lumen.zeros([4, 5, 3]))
    steps = [s["primitive"] for s in lumen.graph_compiler.Plan(graph, "mps", fuse=False).steps()]
    assert steps == ["transpose", "transpose", "dot_general"]
    graph = lumen.make_graph(lambda x: F.sum(x, -1))(lumen.zeros([4, 50_000]))
    plan = lumen.graph_compiler.Plan(graph, "mps")
    offset, size = plan.steps()[0]["scratch"]
    assert size > 0 and plan.workspace_bytes >= offset + size
    x = lumen.ones([4, 50_000], device="mps")
    assert plan.run([x])[0].tolist() == [50_000.0] * 4


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_programs_run_in_their_dtypes(device, triton_on_cuda):
    """Every op computes in its traced dtype: bfloat16 math (exp, log, sqrt,
    tanh, logistic, softmax) has no MPS or Triton kernel and raises there (the CPU's
    reference rounds it once); a dot accumulates in
    its required accum_dtype (float32 for 16-bit floats, written in the
    program; `@` always float32; sum and mean in their input's dtype). A
    dot returns its required output_dtype (`@`: its inputs'), a sum its
    accum_dtype."""
    try:
        x = lumen.from_numpy(rand(4, 8)).to(device).to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    for f in (F.exp, F.sqrt, lambda a: F.softmax(a, -1), F.tanh):
        if device not in ("mps", "cuda"):
            assert lumen.compile(f)(x).dtype == "bfloat16"
            continue
        with pytest.raises(RuntimeError, match="convert to float32"):
            lumen.compile(f)(x)
    out = lumen.compile(lambda a: F.exp(a.float()))(x)
    assert out.dtype == "float32"
    with pytest.raises(TypeError):
        prims.dot_general(x, x, (((1,), (1,)), ((), ())))
    w = lumen.from_numpy(rand(8, 3, seed=1)).to(device).to(dtype="bfloat16")
    dot = lambda a, b, d, o: prims.dot_general(a, b, (((1,), (0,)), ((), ())), d, o)  # noqa: E731
    a, b = (lumen.to_numpy(t.to(dtype="float32")).astype(np.float64) for t in (x, w))
    wide = lumen.compile(lambda a, b: dot(a, b, "float32", "float32"))(x, w)
    assert wide.dtype == "float32"
    np.testing.assert_allclose(lumen.to_numpy(wide), a @ b, rtol=1e-6, atol=1e-6)
    narrow = lumen.compile(lambda a, b: dot(a, b, "float32", "bfloat16"))(x, w)
    assert narrow.dtype == "bfloat16"
    np.testing.assert_allclose(lumen.to_numpy(narrow.to(dtype="float32")), a @ b, rtol=2**-8, atol=2**-8)
    if device == "cuda":
        with pytest.raises(Exception, match="accumulates in float32 or float16"):
            lumen.compile(lambda a, b: dot(a, b, "bfloat16", "bfloat16"))(x, w)
    else:
        assert lumen.compile(lambda a, b: dot(a, b, "bfloat16", "bfloat16"))(x, w).dtype == "bfloat16"
    with pytest.raises(ValueError, match="output_dtype"):
        lumen.compile(lambda a, b: dot(a, b, "bfloat16", "float32"))(x, w)
    graph = lumen.make_graph(lambda a, b: F.matmul(a, b, "float32", a.dtype))(x, w)
    assert "accum_dtype=f32 output_dtype=bf16" in str(graph)
    assert lumen.compile(lambda a, b: F.matmul(a, b, "float32", a.dtype))(x, w).dtype == "bfloat16"
    graph = lumen.make_graph(lambda a: F.sum(a, -1))(x)
    assert [n["primitive"] for n in graph.nodes()] == ["reduce_sum"]
    assert "accum_dtype=bf16" in str(graph)
    assert lumen.compile(lambda a: F.sum(a, -1))(x).dtype == "bfloat16"
    assert lumen.compile(lambda a: F.mean(a))(x).dtype == "bfloat16"
    total = lumen.compile(lambda a: F.sum(a.float(), -1))(x)
    assert total.dtype == "float32"
    np.testing.assert_allclose(lumen.to_numpy(total), a.sum(-1), rtol=1e-6, atol=1e-6)
    back = lumen.compile(lambda a: F.sum(a.float(), -1).bfloat16())(x)
    assert back.dtype == "bfloat16"
    np.testing.assert_allclose(lumen.to_numpy(back.to(dtype="float32")), a.sum(-1), rtol=2**-8, atol=2**-8)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_concatenate_runs_unfused(device, triton_on_cuda):
    """With fusion off, a concatenate (alone a fusion on MPS) still runs."""
    try:
        a = lumen.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(lambda a: prims.concatenate([a, a * 2.0], 1))(a)
    plan = lumen.graph_compiler.Plan(graph, device, fuse=False)
    (out,) = plan.run([a])
    np.testing.assert_array_equal(lumen.to_numpy(out), np.concatenate([lumen.to_numpy(a), 2 * lumen.to_numpy(a)], 1))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_cpu_inside_a_compiled_function(device, triton_on_cuda):
    """``x.cpu()`` (and ``x.to("cpu")``, with a dtype too) inside a compiled
    function, as in PyTorch: a traced tensor has a device (the plan's, or
    ``cpu`` once on the host), and ``cpu()`` of a CPU tensor is the tensor
    itself (assigning it assigns the argument); another device's is copied
    to the host, its own tensor, returned as a CPU tensor."""
    devices = []

    def f(x):
        """Moves values to the host inside the trace and records each one's device."""
        y = x * 2.0
        h = y.cpu()
        devices.extend([x.device, h.device, (h + 1.0).device, h.to(device).device])
        return h, F.sum(h + 1.0).to("cpu"), x.to("cpu", "float16"), x.to(device="cpu"), y, x.cpu().add_(0.5), x

    try:
        g = lumen.compile(f, device=device)
        first = g(lumen.tensor([1.0, 2.0]))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert devices == [f"{device}:0", "cpu:0", "cpu:0", f"{device}:0"]
    h, total, half, moved, y, added, x = first
    assert all(t.device == "cpu:0" for t in (h, total, half, moved, added))
    assert y.device == x.device == f"{device}:0"
    assert half.dtype == "float16"
    assert [h.tolist(), total.tolist(), half.tolist()] == [[2.0, 4.0], [8.0], [1.0, 2.0]]
    expected = [1.5, 2.5] if device == "cpu" else [1.0, 2.0]
    assert added.tolist() == [1.5, 2.5] and moved.tolist() == x.tolist() == expected


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_cpu_inside_a_compiled_function_transfers(device, triton_on_cuda):
    """``.cpu()`` inside a compiled MPS function splits it into stages: the
    ops before it on the device, then its copy to the host (``to_host``:
    ``lumen::copy_d2h``, its ``lumen::wait`` inside), then the ops reading
    it on the host, compiled for the CPU, as PyTorch runs ops on CPU
    tensors (no kernels), the result a CPU tensor. An op reading host and
    device values raises PyTorch's error. The gradient comes back to the
    device (``to_device``)."""

    def mid(x):
        """Device work, a copy to the host, then host work."""
        return (x * 3.0).cpu() * 2.0 + 1.0

    x = lumen.tensor([1.0, 2.0])
    try:
        f = lumen.compile(mid, device=device)
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            y = f(x)
        events = prof.events()
        by_id = {e["id"]: e for e in events}
        assert [e["name"] for e in events if e["kind"] == "gpu"] == ["copy_h2d", "mul"]
        (copy,) = [e for e in events if e["name"] == "lumen::copy_d2h"]
        names = []
        while copy.get("parent") in by_id:
            copy = by_id[copy["parent"]]
            names.append(copy["name"])
        assert "to_host" in names
    else:
        y = f(x)
    assert y.device == "cpu:0" and y.tolist() == [7.0, 13.0]
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        lumen.compile(lambda x: x.cpu() + x, device=device)(x)
    grad = lumen.compile(lumen.grad(lambda x: F.sum((x * 3.0).cpu() * 5.0).to(device)), device=device)(x)
    assert grad.device == f"{device}:0" and grad.tolist() == [15.0, 15.0]
