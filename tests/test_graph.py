"""Compiled execution: tracing with lumen.compile, the torch-like API on
traced tensors, and the strict primitives (lumen/graph/)."""

import dataclasses
import re
import warnings

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)


def run(fn, *args):
    """``lumen.compile(fn)(*args)`` with the results as NumPy arrays."""
    out = lumen.compile(fn)(*args)
    return tuple(map(lumen.to_numpy, out)) if isinstance(out, tuple) else lumen.to_numpy(out)


def rand(*shape, dtype=np.float32, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


# ---------------------------------------------------------------------
# tracing
# ---------------------------------------------------------------------


def test_graph_text():
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x), -1))(lumen.zeros([2, 3]))
    assert str(graph) == (
        "{ lambda %0:f32[2,3]. let\n"
        "    %1:f32[2,3] = exp %0\n"
        "    %2:f32[2] = reduce_sum[axes=(1,) accum_dtype=f32] %1\n"
        "  in (%2) }"
    )


def test_traces_once_per_signature():
    traces = []

    @lumen.compile
    def f(x, scale):
        traces.append(x.shape)
        return x * scale

    f(lumen.ones([2]), 2.0)
    f(lumen.ones([2]), 2.0)
    assert len(traces) == 1
    assert lumen.to_numpy(f(lumen.ones([3]), 2.0)).tolist() == [2.0] * 3
    # A float is a runtime scalar: another value, the same plan.
    assert lumen.to_numpy(f(lumen.ones([3]), 3.0)).tolist() == [3.0] * 3
    assert len(traces) == 2
    # Weakly typed: it takes the tensor's dtype.
    assert f(lumen.ones([3], dtype="float64"), 3.0).dtype == "float64"
    assert len(traces) == 3
    # An int is static, part of the key.
    f(lumen.ones([3]), 2)
    f(lumen.ones([3]), 3)
    assert len(traces) == 5


def test_multiple_outputs_and_passthrough():
    x = lumen.tensor([1.0, 2.0])
    a, b = run(lambda x: (x, x + 1), x)
    assert a.tolist() == [1.0, 2.0] and b.tolist() == [2.0, 3.0]


def test_ops_need_a_trace():
    with pytest.raises(RuntimeError, match="while tracing"):
        prims.full((2,), 1.0, "float32")
    with pytest.raises(TypeError):
        lumen.tensor([1.0]) + 1  # eager tensors have no ops


def test_data_dependent_control_flow_is_rejected():
    def f(x):
        return x if F.sum(x) > 0 else -x

    with pytest.raises(TypeError, match="control flow"):
        lumen.compile(f)(lumen.ones([2]))


def test_must_return_traced_tensors():
    with pytest.raises(TypeError):
        lumen.compile(lambda x: 1.0)(lumen.ones([2]))


def test_factories_record_primitives_while_tracing():
    def f(x):
        return x + lumen.ones([3]) + lumen.full([2, 3], 2.0) + lumen.arange(3) + lumen.zeros([3])

    x = lumen.zeros([2, 3])
    assert "iota" in str(lumen.make_graph(f)(x))
    out = run(f, x)
    assert out.dtype == np.float32
    assert out.tolist() == [[3.0, 4.0, 5.0]] * 2
    assert run(lambda: lumen.full([2], True)).tolist() == [True, True]  # no tensor inputs: on the CPU
    assert isinstance(lumen.ones([2]), lumen.Tensor)  # eager outside a trace


# ---------------------------------------------------------------------
# primitives are strict
# ---------------------------------------------------------------------


def test_prims_do_not_broadcast_or_promote():
    x, y = lumen.zeros([2, 3]), lumen.zeros([3])
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, y)
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, lumen.zeros([2, 3], dtype="float64"))
    with pytest.raises(ValueError, match="floating-point"):
        lumen.compile(prims.exp)(lumen.zeros([2], dtype="int32"))


def test_prims_compose():
    x = rand(2, 3)

    def f(x):
        m = prims.broadcast_in_dim(prims.reduce_max(x, (1,)), (2, 3), (0,))
        i = prims.iota("float32", (2, 3), 1)
        return prims.select(prims.lt(x, m), prims.add(x, i), m)

    m = x.max(1, keepdims=True)
    np.testing.assert_array_equal(run(f, lumen.from_numpy(x)), np.where(x < m, x + np.arange(3, dtype=np.float32), m))


# ---------------------------------------------------------------------
# the torch-like API
# ---------------------------------------------------------------------


def test_no_implicit_dtype_changes():
    i, f = lumen.tensor([1, 2]), lumen.tensor([1.0, 2.0])
    half = lumen.tensor([1.0], dtype="float16")
    # Python scalars take the tensor's dtype.
    assert run(lambda i: i * 2, i).dtype == np.int64
    assert run(lambda f: f * 2, f).dtype == np.float32
    assert run(lambda h: h + 2.5, half).dtype == np.float16
    assert run(lambda b: b == True, lumen.tensor([True, False])).tolist() == [True, False]  # noqa: E712
    # Sums stay in the tensor's dtype (integers wrap): convert for another.
    u8 = lumen.tensor([200, 100], dtype="uint8")
    assert run(lambda x: F.sum(x), u8).tolist() == 44
    assert run(lambda x: F.sum(x.long()), u8).tolist() == 300
    # Explicit conversions are how dtypes change.
    assert run(lambda i: F.exp(i.float()), i).dtype == np.float32
    assert run(lambda i, f: i.to(f.dtype) + f, i, f).tolist() == [2.0, 4.0]
    assert run(lambda i: F.softmax(i.float(), 0), i).dtype == np.float32
    errors = [
        (lambda i, f: i + f, (i, f), "dtypes float32 and int64"),
        (lambda h, f: F.maximum(h, f), (half, lumen.tensor([0.0])), "dtypes float16 and float32"),
        (lambda f, h: F.where(f > 0, f, h), (lumen.tensor([1.0]), half), "dtypes float16 and float32"),
        (
            lambda h, f: h @ f,
            (lumen.zeros([2, 2], dtype="float16"), lumen.zeros([2, 2])),
            "dtypes float16 and float32",
        ),
        (lambda i: i + 1.5, (i,), "int64 tensor and the float 1.5"),
        (lambda b: b + 1, (lumen.tensor([True]),), "bool tensor and the int 1"),
        (lambda i: i / 2, (i,), "true division"),
        (lambda i: F.exp(i), (i,), "exp needs a floating-point tensor"),
        (lambda i: F.softmax(i, 0), (i,), "softmax needs a floating-point tensor"),
    ]
    for fn, args, message in errors:
        with pytest.raises(TypeError, match=message):
            lumen.compile(fn)(*args)


def test_broadcasting():
    a, b = rand(4, 1, 3), rand(5, 1, seed=1)
    np.testing.assert_allclose(
        run(lambda a, b: a * b - a, lumen.from_numpy(a), lumen.from_numpy(b)), a * b - a, rtol=1e-6
    )
    with pytest.raises(RuntimeError, match="broadcastable"):
        lumen.compile(lambda a, b: a + b)(lumen.zeros([2]), lumen.zeros([3]))


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((1, 2, 3), (4, 3, 2))],
)
def test_matmul(a_shape, b_shape):
    a, b = rand(*a_shape), rand(*b_shape, seed=1)
    out = run(lambda a, b: a @ b, lumen.from_numpy(a), lumen.from_numpy(b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def test_matmul_folds_a_batch_times_a_matrix():
    """A batch times a matrix (``x @ w``, a weight) is one matmul of the
    batch's rows, as torch folds it: the matrix read as it is, never
    broadcast over the batch (a copy); its gradient one matmul too, not a
    sum of the batch's."""
    m = lambda *s: lumen.empty(list(s), device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, w: x @ w)(m(2, 3, 4), m(4, 5))
    assert "broadcast_in_dim" not in str(g) and "f32[6,5] = dot_general" in str(g), g
    grad = lumen.make_graph(lumen.grad(lambda x, w: F.sum(x @ w), 1))(m(2, 3, 4), m(4, 5))
    assert str(grad).count("dot_general") == 1 and "reduce_sum" not in str(grad), grad


def test_reductions():
    x = rand(2, 3, 4)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.sum(x), t), x.sum(), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.sum(x, (0, -1), keepdim=True), t), x.sum((0, 2), keepdims=True), rtol=1e-5
    )
    np.testing.assert_allclose(run(lambda x: F.mean(x, 1), t), x.mean(1), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.amax(x, -1), t), x.max(-1))
    np.testing.assert_array_equal(run(lambda x: F.max(x), t), x.max())
    with pytest.raises(RuntimeError, match="floating point"):
        lumen.compile(lambda x: F.mean(x))(lumen.tensor([1, 2]))


def test_softmax_and_friends():
    x = rand(3, 5)
    e = np.exp(x - x.max(-1, keepdims=True))
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.softmax(x, -1), t), e / e.sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: F.sum(F.softmax(x, dim=0), 0), t), np.ones(5), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.log_softmax(x, 1), t), np.log(e / e.sum(-1, keepdims=True)), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(run(lambda x: F.sigmoid(x), t), 1 / (1 + np.exp(-x)), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.relu(x), t), np.maximum(x, 0))


def test_comparisons_and_where():
    x, y = rand(6), rand(6, seed=1)
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        np.testing.assert_array_equal(run(lambda a, b: getattr(a, op)(b), tx, ty), getattr(x, op)(y))
    np.testing.assert_array_equal(run(lambda a, b: F.where(a > b, a, 0.0), tx, ty), np.where(x > y, x, 0.0))
    np.testing.assert_array_equal(run(F.minimum, tx, ty), np.minimum(x, y))
    nan = lumen.tensor([float("nan"), 1.0])
    assert np.isnan(run(F.maximum, nan, lumen.tensor([0.0, 0.0]))[0])


def test_shape_methods():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x.reshape(4, -1): x.reshape(4, -1),
        lambda x: x.view(-1): x.reshape(-1),
        lambda x: x.flatten(1): x.reshape(2, 12),
        lambda x: x.permute(2, 0, 1): x.transpose(2, 0, 1),
        lambda x: x.transpose(0, -1): x.swapaxes(0, -1),
        lambda x: x.mT: x.swapaxes(-2, -1),
        lambda x: x.unsqueeze(1).squeeze(): x,
        lambda x: x.unsqueeze(0).expand(5, -1, -1, -1): np.broadcast_to(x, (5, 2, 3, 4)),
    }
    for fn, expected in cases.items():
        np.testing.assert_array_equal(run(fn, t), expected)
    with pytest.raises(RuntimeError, match="invalid for input of size 24"):
        lumen.compile(lambda x: x.reshape(5, -1))(t)
    with pytest.raises(IndexError, match="Dimension out of range"):
        lumen.compile(lambda x: F.sum(x, 3))(t)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_results_on_input_device(device):
    try:
        x = lumen.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    y = lumen.compile(lambda x: x @ x + 1)(x)
    assert y.device == str(x.device)
    assert lumen.to_numpy(y).tolist() == [[8.0, 11.0], [16.0, 23.0]]


@pytest.mark.mps
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mps_kernels_match_cpu(dtype):
    """Compiled functions on MPS (Metal kernels) agree with the CPU."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))

    def block(x, wq, wk, w1):
        scores = (x @ wq) @ (x @ wk).t() * 0.25
        # Softmax in float32 (bfloat16 has no exp).
        h = x + F.softmax(scores.float(), -1).to(dtype=x.dtype) @ x
        return F.mean(F.relu(h @ w1), -1), F.amax(h, 0), F.sum(F.where(h > 0, h, -h))

    arrays = [rand(8, 16, seed=1), rand(16, 16, seed=2), rand(16, 16, seed=3), rand(16, 32, seed=4)]
    cpu = [lumen.from_numpy(a) for a in arrays]
    fn = lumen.compile(lambda *xs: tuple(o.float() for o in block(*(x.to(dtype) for x in xs))))
    expected = [lumen.to_numpy(o) for o in fn(*cpu)]
    actual = fn(*(t.to("mps") for t in cpu))
    tol = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}[dtype]
    for e, a in zip(expected, actual):
        assert a.device == "mps"
        np.testing.assert_allclose(lumen.to_numpy(a), e, rtol=tol, atol=tol)


def test_plan():
    graph = lumen.make_graph(lambda x: F.tanh((F.exp(x) + 1).reshape(-1)))(lumen.zeros([16, 16]))
    plan = lumen.graph.Plan(graph)
    # exp writes the output's memory, add and tanh write over it in place
    # (each operand dies there); the workspace holds the broadcast 1 (1 KiB)
    # and the 0-d constant; the reshape costs nothing.
    assert "reshape" not in str(plan)
    assert plan.workspace_bytes == 1024 + 4, plan
    x = rand(16, 16)
    np.testing.assert_allclose(
        lumen.to_numpy(plan.run([lumen.from_numpy(x)])[0]), np.tanh(np.exp(x) + 1).ravel(), rtol=1e-6
    )
    with pytest.raises(ValueError, match="must be f32"):
        plan.run([lumen.zeros([16, 16], dtype="float64")])


def test_graph_and_plan_describe_themselves():
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x * 2.0), -1))(lumen.zeros([2, 3]))
    nodes = graph.nodes()
    assert [n["primitive"] for n in nodes] == ["full", "broadcast_in_dim", "mul", "exp", "reduce_sum"]
    assert nodes[-1]["text"] == "reduce_sum[axes=(1,) accum_dtype=f32]" and nodes[-1]["fusion"] is None
    assert graph.inputs() == [0] and graph.outputs() == [nodes[-1]["output"]]
    steps = lumen.graph.Plan(graph).steps()
    assert steps[-1]["output"] == ("out0", "float32", [2])
    assert steps[2]["inputs"][0] == ("in0", "float32", [2, 3])


@pytest.mark.mps
def test_mps_plan_steps_and_profiled_kernels():
    try:
        x = lumen.ones([64, 2048], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    from lumen.profiler import ProfilerActivity, profile

    # Rows long enough for a threadgroup each.
    graph = lumen.make_graph(lambda x: F.sum(F.tanh(x * 2.0 + 1.0), -1))(x)
    plan = lumen.graph.Plan(graph, "mps")
    # One step: the reduction fused with the ops computing its input.
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


@pytest.mark.mps
@pytest.mark.parametrize("n", [16, 64])
def test_short_rows_reduce_several_a_threadgroup(n):
    """Rows too short for a threadgroup each (an attention head's 16, a
    row of 64) are reduced as grouped lanes, several rows a threadgroup:
    the CPU's sums, the same each run."""
    x = rand(4096, n)
    try:
        X = lumen.from_numpy(x).to("mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lambda x: F.sum(F.tanh(x * 2.0 + 1.0), -1)  # noqa: E731
    (step,) = lumen.graph.Plan(lumen.make_graph(f)(X), "mps").steps()
    assert "reduce_grouped" in step["fusion"]["source"]
    got = [lumen.to_numpy(lumen.compile(f)(X)) for _ in range(2)]
    np.testing.assert_array_equal(got[0], got[1])
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(lumen.from_numpy(x)))
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-5)


def test_dump_graph(tmp_path):
    f = lumen.compile(lambda x, w: F.relu(x @ w))
    with pytest.raises(RuntimeError, match="call <lambda> first"):
        f.dump_graph(tmp_path / "none.html")
    f(lumen.zeros([2, 3]), lumen.zeros([3, 4]))
    data = f.dump_graph(tmp_path / "graph.html", json_path=tmp_path / "graph.json")
    page = (tmp_path / "graph.html").read_text()
    assert page.startswith("<!doctype html>") and "<title>&lt;lambda&gt; · lumen graph</title>" in page
    assert data["inputs"] == ["f32[2,3]", "f32[3,4]"] and data["device"] == "cpu"
    assert [n["label"] for n in data["views"]["traced"]["nodes"] if n["kind"] == "node"] == [
        "dot_general",
        "full",
        "broadcast_in_dim",
        "max",
    ]
    # Each node's line of the program, its file's text with it: the
    # lambda's, here.
    lines = {tuple(src) for n in data["views"]["traced"]["nodes"] if n["kind"] == "node" for src in n["sources"]}
    ((file, line),) = lines
    assert file == __file__ and "F.relu(x @ w)" in data["code"][file].splitlines()[line - 1]
    assert {tuple(src) for n in data["views"]["fused"]["nodes"] for src in n.get("sources", [])} == lines
    # Another signature, traced for the dump.
    data = f.dump_graph(
        tmp_path / "f16.html", lumen.zeros([5, 3], dtype="float16"), lumen.zeros([3, 4], dtype="float16")
    )
    assert data["inputs"] == ["f16[5,3]", "f16[3,4]"]


# ---------------------------------------------------------------------
# specs and making tensors by running functions (JAX's design)
# ---------------------------------------------------------------------


def test_meta_tensors_have_no_data():
    x = lumen.empty([2, 3], device="meta")
    assert (x.device, x.shape, x.dtype) == ("meta", [2, 3], "float32")
    assert repr(x) == "Tensor(shape=[2, 3], dtype=f32, device=meta)"
    assert lumen.ones([2, 3], device="meta").device == "meta"  # nothing filled
    assert lumen.tensor([1.0, 2.0]).to("meta").shape == [2]  # data dropped
    assert x.transpose(0, 1).contiguous().shape == [3, 2]  # views, no copies
    for read in (x.tolist, lambda: x.to("cpu"), lambda: x[0, 0], lambda: lumen.to_numpy(x), x.__dlpack__):
        with pytest.raises(RuntimeError, match="no data"):
            read()


def test_tracing_and_running_on_meta_tensors():
    def f(x, w):
        return F.relu(x @ w), F.sum(x, -1)

    x, w = lumen.empty([8, 4], device="meta"), lumen.empty([4, 16], device="meta")
    assert "dot_general" in str(lumen.make_graph(f)(x, w))
    # A compiled function runs nothing on meta tensors: meta results, of
    # the types real ones would have (shape inference).
    y, s = lumen.compile(f)(x, w)
    assert (y.device, y.shape, s.shape) == ("meta", [8, 16], [8])
    assert lumen.compile(lambda: lumen.arange(5), device="meta")().shape == [5]


def test_compiled_init_runs_once_per_call(tmp_path):
    calls = []

    def init():
        calls.append(1)
        return lumen.full([2, 3], 0.5) + lumen.arange(3)

    make = lumen.compile(init)
    a = make().clone()
    b = make()
    assert calls == [1]  # traced once; each call runs the plan
    # Results are views of the function's workspace, which each call reuses.
    assert a.tolist() == b.tolist() == [[0.5, 1.5, 2.5]] * 2 and b.shares_storage_with(make())
    data = make.dump_graph(tmp_path / "init.html")
    assert data["inputs"] == [] and data["device"] == "cpu"


@pytest.mark.mps
def test_compiled_init_on_device(tmp_path):
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    make = lumen.compile(lambda: (lumen.ones([4, 8]) * 2.0, lumen.arange(8)), device="mps")
    w, i = make()
    assert (w.device, i.device) == ("mps", "mps")
    assert w.tolist() == [[2.0] * 8] * 4 and i.tolist() == list(map(float, range(8)))
    # Tensor arguments with data are copied in, from any device.
    assert lumen.compile(lambda x: x + 1.0, device="mps")(lumen.zeros([2])).tolist() == [1.0, 1.0]
    # Profiled on the device it creates tensors on.
    data = make.dump_graph(tmp_path / "init.html")
    assert data["device"] == "mps" and any(n.get("kernels") for n in data["views"]["fused"]["nodes"])
    # Meta tensors trace a dump with no data; it is profiled on `device`.
    f = lumen.compile(lambda x, w: x @ w)
    meta = lumen.empty([2, 4], device="meta"), lumen.empty([4, 3], device="meta")
    data = f.dump_graph(tmp_path / "f.html", *meta, device="mps")
    assert data["device"] == "mps" and data["inputs"] == ["f32[2,4]", "f32[4,3]"]
    assert any(n.get("kernels") for n in data["views"]["fused"]["nodes"])


# ---------------------------------------------------------------------
# memory: parameters, the workspace and planned scratch (XLA buffer
# assignment)
# ---------------------------------------------------------------------


class Affine(lumen.nn.Module):
    w: lumen.Tensor
    b: lumen.Tensor

    def __call__(self, x):
        return x @ self.w + self.b


class Stack(lumen.nn.Module):
    """Submodules in a list, a weight shared by two of them, and a static
    field."""

    layers: list
    scale: float

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) * self.scale
        return x


def meta(*shape):
    return lumen.empty(list(shape), device="meta")


def test_modules_are_frozen_dataclasses_of_weights():
    shared = meta(4)
    m = Stack([Affine(meta(4, 4), shared), Affine(meta(4, 4), shared)], 0.5)
    assert [n for n, _ in m.named_parameters()] == ["layers.0.w", "layers.0.b", "layers.1.w"]
    assert m.parameters()[1] is shared
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.scale = 1.0
    with pytest.raises(ValueError, match="compile it first"):
        m.load_state_dict({n: lumen.zeros(t.shape) for n, t in m.named_parameters()})
    with pytest.raises(KeyError, match="missing"):
        m.load_state_dict({})


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_module_weights_are_placed_once_and_shared(device):
    """A module argument's tensors are the function's weights: placed
    (zeroed) on the device when it compiles (a call on meta tensors, which
    runs nothing), filled by load_state_dict, shared by every function
    taking the module; each call copies the tensor arguments alone."""
    shared = meta(4)
    model = Stack([Affine(meta(4, 4), shared), Affine(meta(4, 4), shared)], 0.5)
    f = lumen.compile(lambda model, x: model(x), device=device)
    try:
        y = f(model, meta(2, 4))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert (y.device, y.shape) == ("meta", [2, 4])
    values = {n: rand(*t.shape, seed=i + 1) for i, (n, t) in enumerate(model.named_parameters())}
    model.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    x = rand(2, 4)
    h = (x @ values["layers.0.w"] + values["layers.0.b"]) * 0.5
    expected = (h @ values["layers.1.w"] + values["layers.0.b"]) * 0.5
    np.testing.assert_allclose(lumen.to_numpy(f(model, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    # Another function reads the same memory; a new copy_ reaches it.
    g = lumen.compile(lambda layer, x: layer(x), device=device)
    first = model.layers[0]
    np.testing.assert_allclose(lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2, rtol=1e-5, atol=1e-5)
    first.w.copy_(lumen.from_numpy(values["layers.0.w"] + 1))
    np.testing.assert_allclose(
        lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2 + x.sum(-1, keepdims=True), rtol=1e-5, atol=1e-4
    )
    # Another model of the same structure reuses the plan, with its weights.
    other = Affine(meta(4, 4), meta(4))
    g(other, meta(2, 4))
    other.load_state_dict({"w": lumen.from_numpy(np.eye(4, dtype=np.float32)), "b": lumen.zeros([4])})
    np.testing.assert_allclose(lumen.to_numpy(g(other, lumen.from_numpy(x))), x, rtol=1e-6)
    # Closing over a tensor is an error; a module's weights must be meta.
    with pytest.raises(TypeError, match="cannot close over a tensor"):
        lumen.compile(lambda x: x @ first.w, device=device)(lumen.from_numpy(x))
    with pytest.raises(TypeError, match="whole meta tensors"):
        g(Affine(lumen.zeros([4, 4]), lumen.zeros([4])), lumen.from_numpy(x))


def test_results_are_views_of_the_workspace():
    f = lumen.compile(lambda x: x * 2.0)
    a = f(lumen.ones([3]))
    assert a.tolist() == [2.0] * 3
    b = f(lumen.full([3], 5.0))
    assert a.shares_storage_with(b) and a.tolist() == [10.0] * 3  # overwritten
    assert not a.clone().shares_storage_with(b)


@pytest.mark.mps
def test_mps_plans_put_copies_and_scratch_in_the_workspace():
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    # A contraction out of matmul form: a transpose step, then the matmul.
    dot = lambda a, b: prims.dot_general(a, b, (((0, 2), (2, 0)), ((), ())), a.dtype, a.dtype)  # noqa: E731
    graph = lumen.make_graph(dot)(lumen.zeros([3, 2, 4]), lumen.zeros([4, 5, 3]))
    steps = [s["primitive"] for s in lumen.graph.Plan(graph, "mps", fuse=False).steps()]
    assert steps == ["transpose", "transpose", "dot_general"]
    # A split reduction's partials are scratch in the workspace.
    graph = lumen.make_graph(lambda x: F.sum(x, -1))(lumen.zeros([4, 50_000]))
    plan = lumen.graph.Plan(graph, "mps")
    offset, size = plan.steps()[0]["scratch"]
    assert size > 0 and plan.workspace_bytes >= offset + size
    x = lumen.ones([4, 50_000], device="mps")
    assert plan.run([x])[0].tolist() == [50_000.0] * 4


# ---------------------------------------------------------------------
# indexing and splitting (prims.slice)
# ---------------------------------------------------------------------


def test_indexing_and_splitting_follow_numpy():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x[1]: x[1],
        lambda x: x[:, 1:3]: x[:, 1:3],
        lambda x: x[..., -1]: x[..., -1],
        lambda x: x[0, ..., 1:]: x[0, ..., 1:],
        lambda x: x[-1, -2, -3]: x[-1, -2, -3],
        lambda x: x[:, :, 2:2]: x[:, :, 2:2],
        lambda x: x[:, 5:]: x[:, 5:],
        lambda x: x.narrow(2, 1, 2): x[:, :, 1:3],
        lambda x: x.narrow(-1, -3, 2): x[:, :, 1:3],
    }
    for fn, expected in cases.items():
        out = run(fn, t)
        assert out.shape == expected.shape and np.array_equal(out, expected)
    assert [p.shape for p in lumen.compile(lambda x: x.split(3, 2))(t)] == [[2, 3, 3], [2, 3, 1]]
    assert [p.shape for p in lumen.compile(lambda x: x.split([1, 3], -1))(t)] == [[2, 3, 1], [2, 3, 3]]
    assert [p.shape for p in lumen.compile(lambda x: x.chunk(2, 1))(t)] == [[2, 2, 4], [2, 1, 4]]
    a, b = run(lambda x: x.chunk(2, -1), t)
    assert np.array_equal(a, x[..., :2]) and np.array_equal(b, x[..., 2:])
    errors = [
        (lambda x: x[2], IndexError, "out of bounds"),
        (lambda x: x[0, 0, 0, 0], IndexError, "too many indices"),
        (lambda x: x[..., ...], IndexError, "single ellipsis"),
        (lambda x: x[::2], NotImplementedError, "steps"),
        (lambda x: x[True], TypeError, "indices must be"),
        (lambda x: x.split([1, 1], 2), RuntimeError, "sum exactly to 4"),
        (lambda x: x.narrow(2, 3, 2), IndexError, "narrow"),
    ]
    for fn, error, message in errors:
        with pytest.raises(error, match=message):
            lumen.compile(fn)(t)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_packed_weights_are_one_matmul(device):
    """relu(x @ w1) * (x @ w3) with w1 and w3 packed into one weight: one
    matmul, on MPS the gate its epilogue (its halves a gated pair)."""
    x, w13 = rand(16, 32), rand(32, 128, seed=1)
    try:
        args = [lumen.from_numpy(a).to(device) for a in (x, w13)]
    except RuntimeError as e:
        pytest.skip(str(e))

    def gated(x, w13):
        gate, up = (x @ w13).chunk(2, -1)
        return F.relu(gate) * up

    out = lumen.to_numpy(lumen.compile(gated)(*args))
    h = x @ w13
    np.testing.assert_allclose(out, np.maximum(h[:, :64], 0) * h[:, 64:], rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(gated)(*args)
    steps = [s["primitive"] for s in lumen.graph.Plan(graph, device).steps()]
    if device == "mps":
        assert steps == ["dot_general → max → mul"], steps
    else:
        assert steps.count("dot_general") == 1


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_concatenate(device):
    a, b, c = rand(2, 3, 4), rand(2, 0, 4, seed=1), rand(2, 5, 4, seed=2)
    try:
        args = [lumen.from_numpy(v).to(device) for v in (a, b, c)]
    except RuntimeError as e:
        pytest.skip(str(e))
    out = lumen.to_numpy(lumen.compile(lambda *xs: prims.concatenate(xs, 1))(*args))
    assert np.array_equal(out, np.concatenate([a, b, c], 1))
    with pytest.raises(ValueError, match="all dimensions but 0"):
        lumen.compile(lambda *xs: prims.concatenate(xs, 0))(*args)


class Gated(lumen.nn.Module):
    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1) * (x @ self.w3)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_dots_sharing_an_operand_merge(device):
    """relu(x @ w1) * (x @ w3) with weights w1 and w3: on MPS, one matmul of
    x and a block holding w1 and w3 side by side (XLA's DotMerger, with the
    weights placed together instead of concatenated), the gate its epilogue
    (a gated pair: one kernel)."""
    x, w1v, w3v = rand(16, 32), rand(32, 64, seed=1), rand(32, 64, seed=2)
    model = Gated(meta(32, 64), meta(32, 64))
    f = lumen.compile(lambda model, x: model(x), device=device)
    try:
        f(model, meta(16, 32))
    except RuntimeError as e:
        pytest.skip(str(e))
    model.load_state_dict({"w1": lumen.from_numpy(w1v), "w3": lumen.from_numpy(w3v)})
    out = lumen.to_numpy(f(model, lumen.from_numpy(x)))
    np.testing.assert_allclose(out, np.maximum(x @ w1v, 0) * (x @ w3v), rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(lambda model, x: model(x))(model, lumen.from_numpy(x))
    plan = lumen.graph.Plan(graph, device, parameters=[1, 2], packable=[1, 2])
    steps = [s["primitive"] for s in plan.steps()]
    if device == "mps":
        assert plan.packed == [([1, 2], 1)] and model.w1._placed("mps").shares_storage_with(model.w3._placed("mps"))
        assert len(steps) == 1 and "concatenate" not in steps
        # Named for the dots it computes, in the plan and the profiler.
        assert plan.steps()[0]["label"] == "2x dot_general → max → mul"
        from lumen.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            f(model, lumen.from_numpy(x))
            lumen.mps.synchronize()
        names = {(e["kind"], e["name"]) for e in prof.events()}
        assert {("op", "2x dot_general → max → mul"), ("gpu", "2x dot_general → max → mul")} <= names
    else:
        assert plan.packed == [] and steps.count("dot_general") == 2


class Attention(lumen.nn.Module):
    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        return F.softmax(q @ k.t(), -1) @ v


@pytest.mark.mps
def test_attention_projections_merge_into_one_matmul():
    """q, k and v (x @ wq, x @ wk, x @ wv) as one matmul of x and a block
    of the three weights; the attention (one flash-attention kernel) reads
    them in place. As traced, the matmuls reading q, k and v read them in
    place, strided views of its result (k's transpose read as the dot's
    dimensions)."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    model = Attention(meta(32, 32), meta(32, 32), meta(32, 32))
    f = lumen.compile(lambda model, x: model(x), device="mps")
    f(model, meta(16, 32))
    values = [rand(32, 32, seed=i + 1) / 4 for i in range(3)]
    model.load_state_dict({n: lumen.from_numpy(v) for n, v in zip(("wq", "wk", "wv"), values)})
    x = rand(16, 32)
    q, k, v = (x @ w for w in values)
    s = np.exp(q @ k.T - (q @ k.T).max(-1, keepdims=True))
    expected = (s / s.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(lumen.to_numpy(f(model, lumen.from_numpy(x))), expected, rtol=1e-4, atol=1e-4)
    graph = lumen.make_graph(lambda model, x: model(x))(model, lumen.from_numpy(x))
    # The attention one kernel (flash attention) reading q, k and v where
    # the merged matmul writes them.
    plan = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
    merged, attention = plan.steps()
    assert (merged["label"], attention["label"]) == ("3x dot_general", "flash_attention")
    assert attention["inputs"] == [(merged["output"][0], "float32", [16, 96])]
    # As traced: its matmuls read q, k and v as strided views.
    lumen.config.compiler.flash_attention = False
    try:
        plan = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
    finally:
        lumen.config.compiler.reset()
    steps = plan.steps()
    assert plan.packed == [([1, 2, 3], 1)]
    assert [s["primitive"] for s in steps].count("dot_general") == 3
    assert [s["label"] for s in steps if s["primitive"] == "dot_general"] == [
        "3x dot_general",
        "dot_general",
        "dot_general",
    ]
    assert "slice" not in [s["primitive"] for s in steps]
    views = [v for s in steps for v in s["views"] if v is not None]
    assert views == [(0, [96, 1]), (32, [96, 1]), (64, [96, 1])]


class Heads(lumen.nn.Module):
    """q, k and v projections of ``[batch * seq, dim]`` rows (``nn.Linear``'s
    ``[out, in]`` weights, merged in training too), attended over 4 heads by
    ``F.flash_attention``."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        q, k, v = ((x @ w.t()).reshape(2, 32, 4, 16) for w in (self.wq, self.wk, self.wv))
        return F.flash_attention(q, k, v, is_causal=True).reshape(64, 64)


@pytest.mark.mps
def test_trained_attention_reads_merged_projections_in_place():
    """In training, the attention's forward and backward kernels both read
    q, k and v where the merged matmul writes them (each a strided view of
    its result, read through its slice and reshape): no copy of each,
    though two kernels read it."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    model = Heads(meta(64, 64), meta(64, 64), meta(64, 64))
    grad = lumen.grad(lambda model, x: F.sum(model(x) * model(x)), (0, 1))
    graph = lumen.make_graph(grad)(model, meta(64, 64))
    steps = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3]).steps()
    labels = [s["label"] for s in steps]
    merged = steps[labels.index("3x dot_general")]["output"][0]
    attention = [s for s in steps if s["label"].startswith("flash_attention")]
    assert len(attention) >= 2, labels
    for s in attention:
        assert merged in [b for b, *_ in s["inputs"]], (s["label"], s["inputs"])
    assert not [label for label in labels if label.startswith("slice")], labels
    # Not a gated pair's backward (q, k and v are three: no two of their
    # gradients' dots merged, which would copy two of the block's weights).
    assert not [label for label in labels if "concatenate" in label], labels


class QKV(lumen.nn.Module):
    """Three projections of x, ``nn.Linear``'s way: ``[out, in]`` weights
    read as ``x @ w.t()``."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        return [x @ w.t() for w in (self.wq, self.wk, self.wv)]


class MixedQKV(QKV):
    """:class:`QKV` in mixed precision, by hand: float32 weights cast to
    bfloat16 where the matmuls read them, the result float32."""

    def __call__(self, x):
        return [F.matmul(x.bfloat16(), w.t().bfloat16(), "float32", "float32") for w in (self.wq, self.wk, self.wv)]


def test_gradients_through_a_rounding_are_written_wide():
    """A float32 value rounded to bfloat16 for matmuls (mixed precision's
    weights, ``w.t().bfloat16()``; ``x.bfloat16()``, read by three): its
    gradient the matmuls' writing their float32 accumulator (summed for
    ``x``), not rounded to bfloat16 and cast back: no precision warning, no
    cast to float32. The gradients NumPy's, to bfloat16 operands' rounding."""
    n, d = 16, 32

    def step(model, x):
        F.sum(sum(F.sum(y * y) for y in model(x))).backward()
        return x.grad, model.wq.grad, model.wk.grad, model.wv.grad

    model = MixedQKV(meta(d, d), meta(d, d), meta(d, d))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = lumen.make_graph(step)(model, meta(n, d))
    assert not any(node["text"] == "cast[new_dtype=f32]" for node in graph.nodes())
    values = {k: rand(d, d, seed=i) / 4 for i, k in enumerate(("wq", "wk", "wv"))}
    x = rand(n, d, seed=3)
    f = lumen.compile(step, device="cpu")
    model = MixedQKV(meta(d, d), meta(d, d), meta(d, d))
    f(model, meta(n, d))
    model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
    got = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    ws = [values[k].astype(np.float64) for k in ("wq", "wk", "wv")]
    ys = [x @ w.T for w in ws]
    want = [sum(2 * y @ w for y, w in zip(ys, ws))] + [2 * y.T @ x for y in ys]
    for g, w in zip(got, want):
        assert np.linalg.norm(g - w) / np.linalg.norm(w) < 2e-2


class SwiGLU(lumen.nn.Module):
    """A gated MLP's up projection, ``silu(x @ w1.t()) * (x @ w3.t())``: in
    float32, or (``mixed``) of bfloat16 operands, the gate in float32, the
    result bfloat16 (read back as float32)."""

    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x, mixed=False, gate=False):
        if mixed:
            g = F.matmul(x.bfloat16(), self.w1.t().bfloat16(), "float32", "float32")
            u = F.matmul(x.bfloat16(), self.w3.t().bfloat16(), "float32", "float32")
            z = (g * F.sigmoid(g) * u).bfloat16().float()
        else:
            g, u = x @ self.w1.t(), x @ self.w3.t()
            z = g * F.sigmoid(g) * u
        return (z, g) if gate else z


@pytest.mark.mps
@pytest.mark.parametrize("mixed", [False, True], ids=["float32", "mixed precision"])
@pytest.mark.parametrize("hidden", [256, 344], ids=["even tiles", "ragged"])
def test_gated_pair_is_its_merged_matmul_epilogue(mixed, hidden):
    """Two projections of one input combined elementwise (SwiGLU's gate and
    value): one matmul of their block whose epilogue computes the gate (its
    tile holding each pair's columns side by side, ``PAIRED``), neither
    projection written out; the values the CPU's. One whose gate is an
    output too is not paired (its dots unmerged), its values the CPU's."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d = 300, 128
    values = {"w1": rand(hidden, d) / np.sqrt(d), "w3": rand(hidden, d, seed=1) / np.sqrt(d)}
    x = rand(m, d, seed=2)
    for gate in (False, True):
        out = {}
        for device in ("cpu", "mps"):
            model = SwiGLU(meta(hidden, d), meta(hidden, d))

            def f(model, x):
                return model(x, mixed, gate)

            g = lumen.compile(f, device=device)
            g(model, meta(m, d))
            model.load_state_dict({k: lumen.from_numpy(v.astype(np.float32)) for k, v in values.items()})
            result = g(model, lumen.from_numpy(x))
            out[device] = [lumen.to_numpy(v) for v in (result if gate else [result])]
            if device == "mps":
                graph = lumen.make_graph(f)(model, meta(m, d))
                steps = lumen.graph.Plan(graph, "mps", parameters=[1, 2], packable=[1, 2]).steps()
                dots = [s["label"] for s in steps if "dot_general" in s["label"]]
                if gate:
                    assert len(dots) == 2, dots
                else:
                    assert len(dots) == 1 and dots[0].startswith("2x dot_general → logistic → mul → mul"), dots
                    assert len(steps) == (3 if mixed else 1), [s["label"] for s in steps]
                    # Its lines of the program: both matmuls'.
                    (merged,) = [s for s in steps if s["label"].startswith("2x")]
                    code = open(__file__).read().splitlines()
                    traced = "\n".join(code[line - 1] for file, line in merged["sources"] if file == __file__)
                    assert "self.w1.t()" in traced and "self.w3.t()" in traced, merged["sources"]
        tol = 2e-2 if mixed else 1e-5
        for got, want in zip(out["mps"], out["cpu"]):
            np.testing.assert_allclose(got, want, atol=tol * np.abs(want).max())


class GatedMLP(lumen.nn.Module):
    """A projection, then a gated MLP of bfloat16 matmuls (ReGLU, or
    SwiGLU's gate in float32): the projection so its input needs a
    gradient, as a layer's does in a model."""

    w0: lumen.Tensor
    wg: lumen.Tensor
    wu: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, inp, swiglu):
        x = F.matmul(inp, self.w0.t().bfloat16(), "float32", "bfloat16")
        if swiglu:
            g = F.matmul(x, self.wg.t().bfloat16(), "float32", "float32")
            u = F.matmul(x, self.wu.t().bfloat16(), "float32", "float32")
            h = (g * F.sigmoid(g) * u).bfloat16()
        else:
            h = F.relu(x @ self.wg.t().bfloat16()) * (x @ self.wu.t().bfloat16())
        return F.matmul(h, self.w2.t().bfloat16(), "float32", "bfloat16")


@pytest.mark.mps
@pytest.mark.parametrize("swiglu", [False, True], ids=["reglu", "swiglu"])
def test_gated_pair_backward_merges_its_gradients_dots(swiglu):
    """A gated pair's backward (``gated_backward``): its input's gradient one
    dot of its cotangents side by side and the forward's weight block (K
    twice as long, no copy of the weights), its weights' gradients one dot;
    the cotangents the down projection's gradient GEMM's expanding
    epilogue, written side by side, whatever the gate (SwiGLU's in float32,
    from the rounded gradient widened: an upcast only that epilogue takes).
    The forward the merged dot's paired epilogue, its halves (which the
    backward reads) written by it too. The gradients the CPU's."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    # m apart from 2h: no two of its values alike in size.
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def step(model, x):
        y = model(x.bfloat16(), swiglu).float()
        F.sum(y * y).backward()
        return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

    grads = {}
    for device in ("cpu", "mps"):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step, device=device)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads[device] = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    for got, want in zip(grads["mps"], grads["cpu"]):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step)(model, meta(m, d))
    steps = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    labels = [s["label"] for s in steps]
    # The forward: the pair's epilogue (writing h [m, h], and the halves the
    # backward reads, no kernel of its own slicing them); the weights'
    # gradients: one dot ([2h, d], named as the two it computes); the
    # input's gradient: one dot of the block (K = 2h), written as x's.
    merged = [s for s in steps if s["label"].startswith("2x dot_general")]
    assert sorted(s["output"][2] for s in merged) == sorted([[m, h], [m, d], [2 * h, d]]), labels
    forward = [s for s in merged if s["output"][2] == [m, h]]
    assert len(forward) == 1 and "mul" in forward[0]["label"], labels
    assert not any(s["label"].startswith("slice") and s["output"][2] == [m, h] for s in steps), labels
    expanding = [s for s in steps if "dot_general" in s["label"] and s["label"].endswith("concatenate")]
    assert len(expanding) == 1 and expanding[0]["output"][2] == [m, 2 * h], labels
    assert sum("concatenate" in label for label in labels) == 1, labels


class _GatedMLPFunction(lumen.autograd.Function):
    """``sigmoid(x wg^T) * (x wu^T)`` then ``wd``, in bfloat16 matmuls, its
    backward by hand (``examples/transformer.py``'s): the weights' gradients
    ``g.t() @ x``, transposes of computed values."""

    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        g = F.matmul(x, wg.t().bfloat16(), "float32", "float32")
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        ctx.save_for_backward(x, g, u, wg, wu, wd)
        return F.matmul((F.sigmoid(g) * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    @staticmethod
    def backward(ctx, dy):
        x, g, u, wg, wu, wd = ctx.saved_tensors
        s = F.sigmoid(g)
        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), (s * u).bfloat16(), "float32", "float32")
        du = (dh * s).bfloat16()
        dg = (dh * u * s * (1 - s)).bfloat16()
        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32") + F.matmul(du, wu.bfloat16(), "float32", "float32")
        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")
        return dx.bfloat16(), dwg, dwu, dwd


@pytest.mark.mps
def test_hand_written_gated_backward_merges_as_autodiffs():
    """A gated pair's backward written by hand (a ``Function``, its
    weights' gradients ``g.t() @ x``): merged as autodiff's is, the
    transposes read as the dots' dimensions (no copies): the cotangents
    the down projection's gradient GEMM's expanding epilogue, x's gradient
    one dot of the block, the weights' one dot. The gradients autodiff's."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def autodiff(x, wg, wu, wd):
        g = F.matmul(x, wg.t().bfloat16(), "float32", "float32")
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        return F.matmul((F.sigmoid(g) * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    def step(mlp):
        def f(model, x):
            x = F.matmul(x.bfloat16(), model.w0.t().bfloat16(), "float32", "bfloat16")
            y = mlp(x, model.wg, model.wu, model.w2).float()
            F.sum(y * y).backward()
            return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

        return f

    grads = []
    for mlp in (_GatedMLPFunction.apply, autodiff):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step(mlp), device="mps")
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads.append([lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))])
    for got, want in zip(*grads):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step(_GatedMLPFunction.apply))(model, meta(m, d))
    steps = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    labels = [s["label"] for s in steps]
    expanding = [s for s in steps if "dot_general" in s["label"] and s["label"].endswith("concatenate")]
    assert len(expanding) == 1 and expanding[0]["output"][2] == [m, 2 * h], labels
    assert sum("concatenate" in label for label in labels) == 1, labels
    # The weights' gradients one dot of [dg | du] and x ([2h, d]).
    assert any(s["primitive"] == "dot_general" and s["output"][2] == [2 * h, d] for s in steps), labels
    # No transpose: the projection's (autodiff's) gradient one float32 dot
    # too, its transpose read as the dot's dimensions.
    assert not any("transpose" in label for label in labels), labels


class _SavedNarrowMLP(lumen.autograd.Function):
    """A gated MLP saving its gate and up projections rounded to bfloat16
    (``examples/transformer.py``'s): its backward reads them so."""

    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        s = F.sigmoid(F.matmul(x, wg.t().bfloat16(), "float32", "float32"))
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        ctx.save_for_backward(x, s.bfloat16(), u.bfloat16(), wg, wu, wd)
        return F.matmul((s * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    @staticmethod
    def backward(ctx, dy):
        x, s, u, wg, wu, wd = ctx.saved_tensors
        s, u = s.float(), u.float()
        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), (s * u).bfloat16(), "float32", "float32")
        du = (dh * s).bfloat16()
        dg = (dh * u * s * (1 - s)).bfloat16()
        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32") + F.matmul(du, wu.bfloat16(), "float32", "float32")
        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")
        return dx.bfloat16(), dwg, dwu, dwd


@pytest.mark.mps
def test_values_saved_narrower_are_written_narrower():
    """Values a matmul's epilogue computes that are read outside it only
    rounded narrower (a backward's saved bfloat16 halves of a gated pair):
    the epilogue writes the rounded values, as the program keeps them, not
    the float32 ones for the backward to round. The gradients the CPU's."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def step(model, x):
        x = F.matmul(x.bfloat16(), model.w0.t().bfloat16(), "float32", "bfloat16")
        y = _SavedNarrowMLP.apply(x, model.wg, model.wu, model.w2).float()
        F.sum(y * y).backward()
        return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

    grads = {}
    for device in ("cpu", "mps"):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step, device=device)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads[device] = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    for got, want in zip(grads["mps"], grads["cpu"]):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step)(model, meta(m, d))
    steps = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    (forward,) = [s for s in steps if s["label"].startswith("2x dot_general") and "logistic" in s["label"]]
    assert [o[1:] for o in forward["extra_outputs"]] == [("bfloat16", [m, h])] * 2, forward["label"]


@pytest.mark.mps
@pytest.mark.parametrize("module", [QKV, MixedQKV], ids=["float32", "mixed precision"])
def test_projections_of_several_inputs_each_read_one_block(module):
    """The same weights projecting two inputs (a gradient accumulation's
    passes, each its own activations): one block of them, each input's
    three projections one matmul reading it (its chain, a bfloat16 copy,
    computed once), the values as unmerged."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(model, x1, x2):
        # Read by fusible primitives (not outputs as they are), so they merge.
        return [F.relu(v) for v in [*model(x1), *model(x2)]]

    model = module(meta(64, 64), meta(64, 64), meta(64, 64))
    graph = lumen.make_graph(f)(model, meta(32, 64), meta(32, 64))
    steps = lumen.graph.Plan(graph, "mps", parameters=[2, 3, 4], packable=[2, 3, 4]).steps()
    labels = [s["label"] for s in steps]
    assert labels.count("3x dot_general") == 2, labels
    blocks = {i[0] for s in steps if s["label"] == "3x dot_general" for i in s["inputs"]}
    assert len(blocks) == 3, blocks  # the two inputs, and one block (or its copy)
    values = rand(64, 64), rand(64, 64, seed=1), rand(64, 64, seed=2)
    xs = [lumen.from_numpy(rand(32, 64, seed=k)).to("mps") for k in (3, 4)]
    got = []
    try:
        for merge in (True, False):
            lumen.config.compiler.merge_dots = merge
            model = module(meta(64, 64), meta(64, 64), meta(64, 64))
            g = lumen.compile(f, device="mps")
            g(model, meta(32, 64), meta(32, 64))
            model.load_state_dict({k: lumen.from_numpy(v) for k, v in zip(["wq", "wk", "wv"], values)})
            got.append([lumen.to_numpy(v) for v in g(model, *xs)])
    finally:
        lumen.config.compiler.reset()
    for merged, alone in zip(*got):
        np.testing.assert_array_equal(merged, alone)


@pytest.mark.mps
@pytest.mark.parametrize("module", [QKV, MixedQKV], ids=["float32", "mixed precision"])
def test_trained_projections_merge_into_one_matmul(module):
    """Weights a training step assigns (its optimizer's update, written over
    them in place) merge too, as PyTorch's fused QKV weight: ``[out, in]``
    weights side by side along their first dimension, each a contiguous
    part of the block, updated there in place once no matmul reads the
    block (the planner's block-aware donation). The weights after a few
    steps are the unmerged program's. In mixed precision too: the weights'
    bfloat16 casts one cast of their block."""
    from lumen.profiler import ProfilerActivity, profile

    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))

    def step(model, x, c):
        q, k, v = model(x)
        loss = F.sum((q * k + v) * c)
        loss.backward()
        for w in (model.wq, model.wk, model.wv):
            w.copy_(w - 0.1 * w.grad)
        return loss

    x, c = lumen.from_numpy(rand(16, 32)), lumen.from_numpy(rand(16, 32, seed=9))
    values = {n: lumen.from_numpy(rand(32, 32, seed=i + 1) / 4) for i, n in enumerate(("wq", "wk", "wv"))}
    trained = {}
    for merge in (True, False):
        lumen.config.compiler.merge_dots = merge
        try:
            model = module(meta(32, 32), meta(32, 32), meta(32, 32))
            f = lumen.compile(step, device="mps")
            f(model, meta(16, 32), meta(16, 32))
            model.load_state_dict(values)
            with profile(activities=[ProfilerActivity.MPS]) as prof:
                for _ in range(3):
                    f(model, x, c)
                lumen.mps.synchronize()
        finally:
            lumen.config.compiler.reset()
        kernels = [e["name"] for e in prof.events() if e["kind"] == "gpu"]
        assert ("3x dot_general" in kernels) == merge, kernels
        trained[merge] = [lumen.to_numpy(w._placed("mps")) for w in (model.wq, model.wk, model.wv)]
    for merged, unmerged in zip(trained[True], trained[False]):
        np.testing.assert_allclose(merged, unmerged, rtol=1e-5, atol=1e-6)


class Linear(lumen.nn.Module):
    w: lumen.Tensor

    def __call__(self, x):
        return x @ self.w


@pytest.mark.mps
def test_packed_weights_are_read_in_place_elsewhere():
    """A weight another function packed into a block (a strided view of it)
    is read in place by the matmuls of a function using it alone, at its
    strides: no copy per call."""
    from lumen.profiler import ProfilerActivity, profile

    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    x, w1v, w3v = rand(16, 32), rand(32, 64, seed=1), rand(32, 64, seed=2)
    gated = Gated(meta(32, 64), meta(32, 64))
    run = lumen.compile(lambda model, x: model(x), device="mps")
    run(gated, meta(16, 32))
    gated.load_state_dict({"w1": lumen.from_numpy(w1v), "w3": lumen.from_numpy(w3v)})
    alone = Linear(gated.w1)
    run(alone, lumen.from_numpy(x))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = run(alone, lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(out), x @ w1v, rtol=1e-5, atol=1e-5)
    assert "lumen::to_vec" not in {e["name"] for e in prof.events()}  # no host copy of w1


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_softmax_traces_to_its_primitives(device):
    """softmax traces to its primitives (max, sub, exp, sum, div; no
    primitive of its own): over the last dimension the MPS compiler runs
    them, with the scale before them, as one row kernel (a chain of
    normalization diamonds). Over another dimension too, as kernels."""
    x = rand(8, 300) * 4
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def scaled(t):
        return F.softmax(t * 0.5, -1)

    graph = lumen.make_graph(scaled)(t)
    primitives = [n["primitive"] for n in graph.nodes()]
    assert "softmax" not in primitives and {"reduce_max", "exp", "reduce_sum", "div"} <= set(primitives)
    s = x * 0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(scaled)(t)), e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7
    )
    if device == "mps":
        (step,) = lumen.graph.Plan(graph, "mps").steps()
        assert step["fusion"] is not None and step["label"].startswith("mul")
    e0 = np.exp(x - x.max(0, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(lambda t: F.softmax(t, 0))(t)),
        e0 / e0.sum(0, keepdims=True),
        rtol=1e-5,
        atol=1e-7,
    )


class Norm(lumen.nn.Module):
    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_rms_norm(device):
    """F.rms_norm (torch.nn.functional.rms_norm), in a module too: traced
    as primitives, which the MPS compiler recognizes, as it does an RMS norm
    written by hand, and fuses into one kernel (its reduction inside) with
    the ops computing its input."""
    x, w = rand(8, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def expected(x, w, eps):
        y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        return y * w if w is not None else y

    f = lambda a, b: F.rms_norm(a + 1.0, 300, b, 1e-6)  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected(x + 1, w, 1e-6), rtol=1e-5, atol=1e-6)
    g = lambda a: F.rms_norm(a, [300])  # noqa: E731  (eps: float32's machine epsilon)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(g)(X)), expected(x, None, 2.0**-23), rtol=1e-5, atol=1e-6)
    assert "rms_norm" not in [n["primitive"] for n in lumen.make_graph(f)(X, W).nodes()]
    by_hand = lambda a, b: b * (a / F.sqrt(1e-6 + F.mean(a * a, -1, keepdim=True)))  # noqa: E731
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(by_hand)(X, W)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6
    )
    if device == "mps":
        steps = lambda f: [
            s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(X, W), "mps").steps()
        ]  # noqa: E731
        # One kernel each: a fusion with its reduction inside (a row kernel).
        for g in (f, by_hand):
            (label,) = steps(g)
            assert "reduce_sum" in label and "sqrt" in label, label
        assert steps(f)[0].startswith("add → mul → reduce_sum")
    with pytest.raises(NotImplementedError, match="last dimension"):
        lumen.make_graph(lambda a: F.rms_norm(a, (8, 300)))(X)
    norm = Norm(lumen.empty([300], device="meta"), 1e-6)
    step = lumen.compile(lambda m, a: m(a), device=device)
    step(norm, lumen.empty([8, 300], device="meta"))
    norm.load_state_dict({"weight": W})
    np.testing.assert_allclose(lumen.to_numpy(step(norm, X)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_sqrt(device):
    """sqrt is a primitive (rsqrt is not: write 1 / F.sqrt(x)); on MPS a fused
    division by a fused sqrt stays a sqrt and a division, as traced."""
    x = rand(8, 16) ** 2 + 0.1
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    assert not hasattr(F, "rsqrt") and not hasattr(lumen.graph.TracedTensor, "rsqrt")
    for f, expected in [
        (lambda a: F.sqrt(a), np.sqrt(x)),
        (lambda a: 3.0 / F.sqrt(a), 3 / np.sqrt(x)),
    ]:
        np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(t)), expected, rtol=1e-6)
    if device == "mps":
        (step,) = lumen.graph.Plan(lumen.make_graph(lambda a: 1.0 / F.sqrt(a))(t), "mps").steps()
        source = step["fusion"]["source"]
        assert step["label"].endswith("div") and "rsqrt(" not in source
        assert "Sqrt::apply" in source and "Div::apply" in source
        # A division by the sqrt of a row's sum (a diamond: one row kernel)
        # stays a sqrt, once a row, and a division.
        f = lambda a: a / F.sqrt(F.sum(a, -1, keepdim=True))  # noqa: E731
        (step,) = lumen.graph.Plan(lumen.make_graph(f)(t), "mps").steps()
        source = step["fusion"]["source"]
        assert "Sqrt::apply" in source and "Div::apply" in source and "rsqrt(" not in source


class ScaledNorm(lumen.nn.Module):
    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        return self.weight * (x / F.sqrt(F.mean(x * x, -1, keepdim=True) + self.eps))


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_float_fields_are_runtime_scalars(device):
    """A module's float field is a runtime scalar: another value runs the same
    plan (no trace, no kernel compiled), read by the kernel, not baked in; an
    RMS norm with it is still one kernel on MPS."""
    x, w = rand(8, 300) * 0.3, rand(300, seed=1)
    traces = []

    def forward(m, a):
        traces.append(1)
        return m(a)

    weight = lumen.empty([300], device="meta")
    f = lumen.compile(forward, device=device)
    try:
        f(ScaledNorm(weight, 1e-6), lumen.empty([8, 300], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    ScaledNorm(weight, 1e-6).load_state_dict({"weight": lumen.from_numpy(w)})
    for eps in (1e-6, 0.1, 1.0):
        out = lumen.to_numpy(f(ScaledNorm(weight, eps), lumen.from_numpy(x)))
        expected = w * x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    assert len(traces) == 1
    # One plan, compiled once: each run reads eps from the host, a kernel
    # argument on MPS (no copy to the device, no new plan).
    graph = lumen.make_graph(forward)(ScaledNorm(weight, 0.5), lumen.from_numpy(x))
    plan = lumen.graph.Plan(graph, device, parameters=[2], scalars=[1])
    workspace = lumen.Tensor.empty([plan.workspace_bytes], "uint8", device)
    inputs = [lumen.from_numpy(x).to(device), None, lumen.from_numpy(w).to(device)]
    for eps in (1e-6, 0.1, 1.0):
        inputs[1] = lumen.Tensor.full([], eps, "float32")
        (out,) = plan.run_in(workspace, inputs)
        expected = w * x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        np.testing.assert_allclose(lumen.to_numpy(out), expected, rtol=1e-5, atol=1e-6)
    (step,) = [s for s in plan.steps() if ("s1", "float32", []) in s["inputs"]]
    if device == "mps":
        assert len(plan.steps()) == 1
        assert "constant float &in1" in step["fusion"]["source"]  # eps, by value


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_programs_run_in_their_dtypes(device):
    """Every op computes in its traced dtype: bfloat16 math (exp, log, sqrt,
    tanh, logistic, softmax) has no kernel and raises; a dot accumulates in
    its required accum_dtype (float32 for 16-bit floats, written in the
    program; `@` always float32; sum and mean in their input's dtype). A
    dot returns its required output_dtype (`@`: its inputs'), a sum its
    accum_dtype."""
    try:
        x = lumen.from_numpy(rand(4, 8)).to(device).to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    for f in (F.exp, F.sqrt, lambda a: F.softmax(a, -1), F.tanh):
        with pytest.raises(ValueError, match="convert to float32"):
            lumen.compile(f)(x)
    out = lumen.compile(lambda a: F.exp(a.float()))(x)
    assert out.dtype == "float32"
    with pytest.raises(TypeError):
        prims.dot_general(x, x, (((1,), (1,)), ((), ())))
    w = lumen.from_numpy(rand(8, 3, seed=1)).to(device).to(dtype="bfloat16")
    dot = lambda a, b, d, o: prims.dot_general(a, b, (((1,), (0,)), ((), ())), d, o)  # noqa: E731
    a, b = (lumen.to_numpy(t.to(dtype="float32")).astype(np.float64) for t in (x, w))
    # Accumulated in float32, written in float32, or rounded once to the
    # operands' bfloat16.
    wide = lumen.compile(lambda a, b: dot(a, b, "float32", "float32"))(x, w)
    assert wide.dtype == "float32"
    np.testing.assert_allclose(lumen.to_numpy(wide), a @ b, rtol=1e-6, atol=1e-6)
    narrow = lumen.compile(lambda a, b: dot(a, b, "float32", "bfloat16"))(x, w)
    assert narrow.dtype == "bfloat16"
    np.testing.assert_allclose(lumen.to_numpy(narrow.to(dtype="float32")), a @ b, rtol=2**-8, atol=2**-8)
    assert lumen.compile(lambda a, b: dot(a, b, "bfloat16", "bfloat16"))(x, w).dtype == "bfloat16"
    with pytest.raises(ValueError, match="output_dtype"):
        lumen.compile(lambda a, b: dot(a, b, "bfloat16", "float32"))(x, w)
    # `@` accumulates floats in float32, its result of the inputs' dtype.
    graph = lumen.make_graph(lambda a, b: a @ b)(x, w)
    assert "accum_dtype=f32 output_dtype=bf16" in str(graph)
    assert lumen.compile(lambda a, b: a @ b)(x, w).dtype == "bfloat16"
    # Sum and mean follow their input's dtype: bfloat16 sums in bfloat16; a
    # float32 sum is written as one (F.sum(x.float())), its cast back the
    # reduction's epilogue.
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


class UpcastNorm(lumen.nn.Module):
    """Llama's RMS norm: normalized in float32, scaled in the input's dtype."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        h = x.float()
        h = h / F.sqrt(F.mean(h * h, -1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype=x.dtype)


@pytest.mark.parametrize("n", [300, 1024, 4096])
def test_upcast_rms_norm_is_one_kernel(n):
    """On MPS an RMS norm in float32 of a bfloat16 input, cast back and
    scaled by its weight, is one row kernel: the cast and the weight run in
    its last pass (an epilogue). Rows of up to 8 elements a thread read x
    once, keeping x.float() (its traced dtype) in registers for the last
    pass. It agrees with the CPU to a bfloat16 rounding."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    weight = lumen.empty([n], dtype="bfloat16", device="meta")
    graph = lumen.make_graph(lambda m, a: m(a))(
        UpcastNorm(weight, 1e-6), lumen.empty([8, n], dtype="bfloat16", device="meta")
    )
    (step,) = lumen.graph.Plan(graph, "mps", parameters=[2], scalars=[1]).steps()
    source = step["fusion"]["source"]
    assert step["label"].endswith("div → cast(float32 -> bfloat16) → broadcast_in_dim → mul"), step["label"]
    cached = n <= 8 * 256
    assert ("float kept0[" in source) == cached
    assert source.count("in0[j]") == (1 if cached else 2), source
    x, w = rand(8, n), rand(n, seed=1)
    out = {}
    for device in ("cpu", "mps"):
        f = lumen.compile(lambda m, a: m(a), device=device)
        norm = UpcastNorm(weight, 1e-6)
        f(norm, lumen.empty([8, n], dtype="bfloat16", device="meta"))
        norm.load_state_dict({"weight": lumen.from_numpy(w).to(dtype="bfloat16")})
        y = f(norm, lumen.from_numpy(x).to(dtype="bfloat16"))
        out[device] = lumen.to_numpy(y.to(dtype="float32"))
    np.testing.assert_allclose(out["mps"], out["cpu"], rtol=2**-7, atol=2**-7)


@pytest.mark.mps
def test_split_reduction_converts_its_input_once():
    """``F.sum(x.float(), -1)`` of bfloat16 rows too long for one launch is two:
    the fused kernel converts each element once and writes float32
    partials; the second, the plain float32 kernel, sums those partials,
    converting nothing (a second convert of float32 values would change
    no value, so it is checked by what each launch runs)."""
    try:
        x = lumen.ones([4, 200_000], device="mps").to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda a: F.sum(a.float(), -1))
    graph = lumen.make_graph(lambda a: F.sum(a.float(), -1))(x)
    (step,) = lumen.graph.Plan(graph, "mps").steps()
    assert step["label"] == "cast(bfloat16 -> float32) → reduce_sum" and step["scratch"] is not None
    source = step["fusion"]["source"]
    assert source.count("convert_value<float>") == 1, source
    assert "device float *out" in source and "device const bfloat *in0" in source, source
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        out = f(x)
        lumen.mps.synchronize()
    first, last = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert first["kernel"] == step["fusion"]["kernel"]
    # The second reads the float32 partials with the plain float32 kernel.
    assert last["kernel"] == "reduce_sum_rows_f32"
    assert last["inputs"] == first["outputs"] and last["inputs"][0][0] == "float32"
    np.testing.assert_array_equal(lumen.to_numpy(out), np.full(4, 200_000, np.float32))


def _layer_norm(a, w):
    xc = a - F.mean(a, -1, keepdim=True)
    return xc / F.sqrt(F.mean(xc * xc, -1, keepdim=True) + 1e-5) * w


@pytest.mark.parametrize(
    "f",
    [
        lambda a, w: a / F.sqrt(F.mean(a * a, -1, keepdim=True) + 1e-6) * w,
        lambda a, w: a / F.sqrt(F.sum(a * a, -1, keepdim=True) + 1e-6) * w,
        lambda a, w: a * (1.0 / F.sqrt(F.mean(a * a, -1, keepdim=True))),
        lambda a, w: (lambda e: e / F.sum(e, -1, keepdim=True))(F.exp(a - F.amax(a, -1, keepdim=True))),
        _layer_norm,
    ],
    ids=["rms mean", "rms sum", "rms reciprocal", "softmax written out", "layer norm"],
)
def test_normalizations_are_one_row_kernel(f):
    """Whatever normalizes rows by reductions of them is a normalization
    diamond, or a chain of them (XLA's SoftmaxRewriterTriton), however it is
    written: one row kernel on MPS, its reductions inside, agreeing with the
    CPU. Nothing matches any one normalization. With no backward to read
    its values of a row (an RMS norm's sqrt(mean + eps)), it writes its
    output alone."""
    x, w = rand(64, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to("mps"), lumen.from_numpy(w).to("mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    (step,) = lumen.graph.Plan(lumen.make_graph(f)(X, W), "mps").steps()
    assert step["fusion"] is not None and "reduce_" in step["label"], step["label"]
    assert step["extra_outputs"] == [], step["extra_outputs"]
    expected = lumen.to_numpy(lumen.compile(f, device="cpu")(lumen.from_numpy(x), lumen.from_numpy(w)))
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected, rtol=1e-5, atol=1e-5)


NORMALIZATIONS = [
    lambda a, w: a / F.sqrt(F.mean(a * a, -1, keepdim=True) + 1e-6) * w,
    lambda a, w: a / F.sqrt(F.sum(a * a, -1, keepdim=True) + 1e-6) * w,
    lambda a, w: a * (1.0 / F.sqrt(F.mean(a * a, -1, keepdim=True))),
    lambda a, w: (lambda e: e / F.sum(e, -1, keepdim=True))(F.exp(a - F.amax(a, -1, keepdim=True))),
    _layer_norm,
]


@pytest.mark.parametrize(
    "f", NORMALIZATIONS, ids=["rms mean", "rms sum", "rms reciprocal", "softmax written out", "layer norm"]
)
def test_normalizations_train_in_one_row_kernel(f):
    """Trained, a normalization's forward is still one row kernel: the
    values of a row its backward reads (an RMS norm's sqrt(mean + eps)) and
    its value before its epilogue (``y`` of ``y * w``, for w's gradient)
    are its outputs too, as a fused RMS norm writes each row's ``rstd``,
    never computed again; the gradients the CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to("mps") for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(lambda a, w, c: F.sum(f(a, w) * c), (0, 1))
    labels = [s["label"] for s in lumen.graph.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()]
    if "sqrt" in str(labels):
        assert sum("sqrt" in label for label in labels) == 1, labels
    got = lumen.compile(grad)(X, W, C)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n, vocab", [(64, 16), (300, 1000), (128, 32000)])
def test_cross_entropy_is_one_row_kernel(n, vocab):
    """``F.cross_entropy`` (``reduction="none"``): each row's ``logsumexp -
    logit[label]``. On MPS one row kernel reads the logits once (an online
    max and sum, and the label's logit picked by a one-hot sum, all one
    loop over the row: none reads another's value), writing a loss a row.
    Trained (``g`` not read from the losses), its gradient ``(exp(x - lse)
    - one_hot) · g`` is the same kernel's second pass over the row (XMA's
    forward-backward kernel), the losses its other output: one kernel,
    nothing of the logits' shape but the gradient written. Losses and
    gradients NumPy's."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    c = rng.standard_normal(n).astype(np.float32)

    def step(x, t, c):
        loss = F.cross_entropy(x, t)
        F.sum(loss * c).backward()
        return loss, x.grad

    try:
        X, T, C = (lumen.from_numpy(a).to("mps") for a in (x, t, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    (forward,) = lumen.graph.Plan(lumen.make_graph(F.cross_entropy)(X, T), "mps").steps()
    assert forward["output"][2] == [n] and "reduce_max" in forward["label"], forward["label"]
    source = forward["fusion"]["source"]
    assert source.count("for (uint c") + source.count("for (uint e = 0") == 1, source
    steps = lumen.graph.Plan(lumen.make_graph(step)(X, T, C), "mps").steps()
    assert [s["output"][2] for s in steps] == [[n, vocab]], [s["label"] for s in steps]
    assert [o[2] for o in steps[0]["extra_outputs"]] == [[n]], steps[0]["label"]
    m = x.max(-1, keepdims=True).astype(np.float64)
    lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
    hit = np.zeros_like(x, dtype=np.float64)
    hit[np.arange(n), t] = 1
    loss, grad = (lumen.to_numpy(v) for v in lumen.compile(step)(X, T, C))
    np.testing.assert_allclose(loss, lse - x[np.arange(n), t], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad, (np.exp(x - lse[:, None]) - hit) * c[:, None], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("rows_read", ["its own", "every row's"])
@pytest.mark.parametrize("n, vocab", [(64, 16), (128, 32000)])
def test_cross_entropy_gradient_reading_the_losses(n, vocab, rows_read):
    """A cross entropy whose gradient's ``g`` reads the losses. Each row's
    its own (``Σ loss²``, ``g = 2·loss``): computed once a row in its row
    kernel after the reductions, the gradient its second pass (one
    kernel). Every row's (``mean(loss)²``, ``g = 2·mean(loss)/n``): not
    known until every row's loss is, so the gradient is a pass after the
    row kernel (and the mean) reading ``lse``, which the row kernel writes
    beside the losses; never ``softmax - one_hot`` written to be scaled
    later. The gradients NumPy's."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    of = {"its own": lambda loss: F.sum(loss * loss), "every row's": lambda loss: F.mean(loss) * F.mean(loss)}

    def step(x, t):
        loss = F.cross_entropy(x, t)
        of[rows_read](loss).backward()
        return loss, x.grad

    try:
        X, T = (lumen.from_numpy(a).to("mps") for a in (x, t))
    except RuntimeError as e:
        pytest.skip(str(e))
    steps = lumen.graph.Plan(lumen.make_graph(step)(X, T), "mps").steps()
    labels = [s["label"] for s in steps]
    if rows_read == "its own":
        assert [s["output"][2] for s in steps] == [[n, vocab]], labels
    else:
        # The row kernel, then the mean (and ``g``, a row each), then the
        # gradient.
        assert [s["output"][2] for s in steps] == [[n], [n], [n, vocab]], labels
        assert [o[2] for o in steps[0]["extra_outputs"]] == [[n]] and "reduce" not in steps[2]["label"], labels
    xr = x.astype(np.float64)
    m = xr.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(xr - m).sum(-1, keepdims=True)))[:, 0]
    hit = np.zeros_like(xr)
    hit[np.arange(n), t] = 1
    loss = lse - xr[np.arange(n), t]
    g = 2 * loss[:, None] if rows_read == "its own" else 2 * loss.mean() / n
    _, grad = (lumen.to_numpy(v) for v in lumen.compile(step)(X, T))
    np.testing.assert_allclose(grad, g * (np.exp(xr - lse[:, None]) - hit), rtol=1e-4, atol=1e-5)


def test_rms_norm_backward_recomputes_its_normalized_input():
    """A training step's RMS norm (its value read forward, its gradients
    taken): its forward row kernel writes its output and each row's
    ``sqrt(mean + eps)``, not ``x / rms`` (``x̂``, w's gradient's): the
    backward, reading ``x`` and ``rms`` anyway, divides again, a value of
    the rows' shape less kept from the forward. The loss and gradients the
    CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)

    def step(x, w, c):
        loss = F.sum(F.rms_norm(x, 300, w) * c)
        loss.backward()
        return loss, x.grad, w.grad

    try:
        X, W, C = (lumen.from_numpy(a).to("mps") for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    steps = lumen.graph.Plan(lumen.make_graph(step)(X, W, C), "mps").steps()
    (forward,) = [s for s in steps if "sqrt" in s["label"]]
    assert [o[2] for o in forward["extra_outputs"]] == [[64, 1]], forward["label"]
    got = lumen.compile(step)(X, W, C)
    want = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "f, kernels",
    [
        (lambda x, w, c: F.sum(F.rms_norm(x, 300, w) * c), 3),
        (lambda x, w, c: F.sum(_layer_norm(x, w) * c), 4),
        (lambda x, w, c: F.sum(F.softmax(x * w, -1) * c), None),
    ],
    ids=["rms norm", "layer norm", "softmax"],
)
def test_normalization_backwards_are_row_kernels(f, kernels):
    """A normalization's backward reduces each row and applies the result
    to the row's values, not one producer's (``dx = g·w / rms - x ·
    Σ_row(…)``): beyond a diamond, a row fusion still, one row kernel for
    x's gradient (a layer norm's two reductions in it), the weight's a sum
    over rows of its own; the forward's kernels besides. The gradients the
    CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to("mps") for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    steps = lumen.graph.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
    if kernels is not None:
        assert len(steps) == kernels, [s["label"] for s in steps]
    got = lumen.compile(grad)(X, W, C)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n", [16, 64])
@pytest.mark.parametrize(
    "f",
    [
        lambda x, w, c: F.sum(F.rms_norm(x, x.shape[-1], w) * c),
        lambda x, w, c: F.sum((_layer_norm(x, w) + w) * c),
        lambda x, w, c: F.sum(F.softmax(x * w, -1) * c),
    ],
    ids=["rms norm", "layer norm", "softmax"],
)
def test_short_rows_are_a_simd_group_each(f, n):
    """Row kernels of short rows (a model's width of 64, an attention
    head's 16) reduce each row in a SIMD group, eight a threadgroup (none
    left idle), forward and backward (its weight's gradient's blocks too,
    each SIMD group's rows combined in order): the CPU's values, the same
    bits each run; rows not a multiple of eight too."""
    x, w, c = rand(1003, n), rand(n, seed=1), rand(1003, n, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to("mps") for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    steps = lumen.graph.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
    rows = [s for s in steps if s["fusion"] and "simd_sum" in s["fusion"]["source"]]
    assert rows, [s["label"] for s in steps]
    got = [[lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)] for _ in range(2)]
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, a, e in zip(*got, want):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_allclose(g, lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "f, sums",
    [
        (lambda x, w, c: F.sum(F.rms_norm(x, 64, w) * c), 1),
        (lambda x, w, c: F.sum((_layer_norm(x, w) + w) * c), 2),
    ],
    ids=["rms norm", "layer norm"],
)
def test_normalization_backwards_sum_weight_gradients_as_they_go(f, sums):
    """A normalization's weight's gradient (``Σ_rows g·y``), a sum over
    rows its backward's row kernel reads the values of: that kernel adds
    them as it goes (a threadgroup a block of rows, each column's sum in
    registers), writing a block's sums each, then one kernel sums those
    (a block at a time, in order): two kernels for the backward, the sums
    the same bits every run, the gradients the CPU's."""
    x, w, c = rand(1024, 64), rand(64, seed=1), rand(1024, 64, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to("mps") for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    steps = lumen.graph.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
    backward = [s for s in steps if s["extra_outputs"] and s["extra_outputs"][0][2] == [64, 64]]
    assert len(backward) == 1, [s["label"] for s in steps]
    assert len(backward[0]["extra_outputs"]) == sums
    # The blocks' sums, each summed by one launch (unsplit: few blocks).
    final = [s for s in steps if s["primitive"] == "reduce_sum"]
    assert len(final) == sums and all(s["inputs"][0][2] == [64, 64] for s in final)
    got = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    again = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, a, e in zip(got, again, want):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_allclose(g, lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


def test_ops_are_functions():
    """The ops are functions in lumen.functional (``import lumen.functional
    as F``), as torch's are; a traced tensor keeps only its operators, layout
    and dtype casts, and lumen itself re-exports no op."""
    ops = ["exp", "log", "sqrt", "tanh", "sigmoid", "relu", "sum", "mean", "amax", "max", "softmax"]
    ops += ["log_softmax", "rms_norm", "matmul", "where", "maximum", "minimum", "add", "mul", "div", "eq", "gt"]
    for name in ops:
        assert callable(getattr(F, name)) and name in F.__all__
        assert not hasattr(lumen.graph.TracedTensor, name), name
        assert not hasattr(lumen, name), name
    for name in ["reshape", "permute", "transpose", "t", "unsqueeze", "flatten", "split", "to", "float", "__add__"]:
        assert hasattr(lumen.graph.TracedTensor, name), name
    assert lumen.functional is F
    # Operators and functions record the same primitives.
    a = lumen.zeros([2, 3])
    by_operator = lumen.make_graph(lambda x: (x + 1) * x @ x.t())(a)
    by_function = lumen.make_graph(lambda x: F.matmul(F.mul(F.add(x, 1), x), x.t(), "float32", "float32"))(a)
    assert str(by_operator) == str(by_function)


class _ScaledMatmul(lumen.nn.Module):
    w: lumen.Tensor
    s: float

    def __call__(self, x):
        return (x @ self.w) * self.s


@pytest.mark.mps
@pytest.mark.parametrize(
    "f, shapes, dtype",
    [
        (lambda x, w, y: F.relu(x @ w + y[0]), [(64, 96), (96, 80), (2, 80)], "float32"),
        (lambda x, w, y: x @ w + y, [(3, 40, 24), (3, 24, 56), (3, 40, 56)], "float32"),
        (lambda x, w, y: x @ w + y, [(3, 40, 24), (24, 56), (3, 40, 56)], "float32"),
        (
            lambda x, w, y: F.relu(F.matmul(x, w, "float32", "float32")).bfloat16(),
            [(64, 96), (96, 80), (64, 80)],
            "bfloat16",
        ),
        (lambda x, w, y: x @ w * F.exp(y), [(64, 32), (32, 48), (64, 48)], "float16"),
        (
            lambda x, w, y: _silu(F.matmul(x, w, "float32", "float32")).to(dtype=x.dtype),
            [(64, 96), (96, 80), (64, 80)],
            "bfloat16",
        ),
    ],
    ids=["bias relu", "batched residual", "folded residual", "relu cast", "times exp(y)", "silu in float32"],
)
def test_matmul_epilogues_fuse(f, shapes, dtype):
    """The elementwise primitives after a matmul (a bias, a residual, an
    activation, a cast, a product with another fused value) run in its
    kernel as it writes each output (``contraction_epilogues``): one step,
    exactly the unfused plan's result."""
    rng = np.random.default_rng(0)
    try:
        args = [lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype=dtype) for s in shapes]
    except RuntimeError as e:
        pytest.skip(str(e))
    (step,) = lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()
    assert step["label"].startswith("dot_general →"), step["label"]
    fused = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
    lumen.config.compiler.contraction_epilogues = False
    try:
        assert len(lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()) == 2
        unfused = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_array_equal(fused, unfused)


@pytest.mark.mps
@pytest.mark.parametrize("m, tiles", [(64, "mid"), (192, "mid"), (16, "small"), (256, "large")])
def test_matmul_tiles_fit_its_rows(m, tiles):
    """A float matmul runs on 128x64 tiles; on 64x64 ones where M leaves 33
    to 64 rows of a last one (M of 64, 192: half a large tile idle, the
    32x32 tiles reading the operands twice as often); on 32x32 where it
    leaves 32 or fewer (or the large tiles are too few to fill the GPU).
    With an epilogue too (its kernel's ``_mid``, ``_small`` variant). The
    results NumPy's."""
    rng = np.random.default_rng(0)
    k, n = 256, 2048
    try:
        a, b = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((m, k), (k, n))
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    product = lumen.to_numpy(a.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(b.to(dtype="float32"))

    def plain(a, b):
        return F.matmul(a, b, "float32", "float32")

    def fused(a, b):
        return F.relu(plain(a, b))

    for f, want in ((plain, product), (fused, np.maximum(product, 0))):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            got = lumen.to_numpy(lumen.compile(f)(a, b))
        (kernel,) = {e["kernel"] for e in prof.events() if e["kind"] == "gpu"}
        ran = "small" if "small" in kernel else "mid" if "mid" in kernel else "large"
        assert ran == tiles, kernel
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)


def _silu(h):
    return h * F.sigmoid(h)


@pytest.mark.mps
def test_matmul_epilogue_stops_at_an_upcast_of_its_rounded_result():
    """A matmul whose result is rounded narrower than it accumulates
    (bfloat16 of float32) keeps a widening cast of it out of its epilogue:
    the epilogue fuses up to it (an add), the cast and what reads it a
    kernel after. Its result wanted wider, its output_dtype says so (one
    kernel). So too a float32 result the epilogue rounds (``.bfloat16()``,
    a backward's rounded gradient) and widens again for a reader that
    widens it itself: the kernel writes the bfloat16 value; widened for
    an output, the cast is the epilogue's (no kernel of its own)."""
    m = lambda *s, d="bfloat16": lumen.empty(list(s), dtype=d, device="meta")  # noqa: E731
    args = (m(64, 96), m(96, 80), m(64, 80, d="float32"))
    labels = lambda f: [s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()]  # noqa: E731
    # The rounding then widening is what the precision check warns of.
    with pytest.warns(UserWarning, match="casts it back"):
        assert labels(lambda x, w, r: (x @ w).float() + r) == [
            "dot_general → cast(float32 -> bfloat16)",
            "cast(bfloat16 -> float32) → add",
        ]
    with pytest.warns(UserWarning, match="casts it back"):
        assert labels(lambda x, w, r: ((x @ w) + 1.0).float() + r) == [
            "dot_general → cast(float32 -> bfloat16) → add",
            "cast(bfloat16 -> float32) → add",
        ]
    assert labels(lambda x, w, r: F.matmul(x, w, "float32", "float32") + r) == ["dot_general → add"]
    rounded = lambda x, w: F.matmul(x, w, "float32", "float32").bfloat16().float()  # noqa: E731
    assert labels(lambda x, w, r: rounded(x, w) * r) == [
        "dot_general → cast(float32 -> bfloat16)",
        "cast(bfloat16 -> float32) → mul",
    ]
    assert labels(lambda x, w, r: rounded(x, w)) == [
        "dot_general → cast(float32 -> bfloat16) → cast(bfloat16 -> float32)"
    ]


@pytest.mark.mps
def test_narrowing_cast_of_a_matmul_is_its_output_dtype():
    """A matmul of bfloat16 operands writing its float32 accumulator as it
    is, read only rounded to bfloat16: the matmul writes bfloat16 (one
    kernel, epilogues or not), the same bits. Read in float32 too, it
    writes float32; rounded to another dtype (float16), the cast stays."""
    rng = np.random.default_rng(0)
    try:
        x, w = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((64, 96), (96, 80))
        )
    except RuntimeError as e:
        pytest.skip(str(e))

    def rounded(x, w):
        y = F.matmul(x, w, "float32", "float32")
        return y.bfloat16()

    both = lambda x, w: (rounded(x, w), F.matmul(x, w, "float32", "float32"))  # noqa: E731

    def dots(f, *args):
        return [s["text"] for s in lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()]

    lumen.config.compiler.contraction_epilogues = False
    try:
        (dot,) = dots(rounded, x, w)
        assert dot.startswith("dot_general") and "output_dtype=bf16" in dot
        # Its lines: the matmul's and the cast's.
        (step,) = lumen.graph.Plan(lumen.make_graph(rounded)(x, w), "mps").steps()
        code = [open(f).read().splitlines()[line - 1].strip() for f, line in step["sources"]]
        assert sorted(code) == ["return y.bfloat16()", 'y = F.matmul(x, w, "float32", "float32")'], code
        assert [t.split("[")[0] for t in dots(both, x, w)] == ["dot_general", "cast"]
    finally:
        lumen.config.compiler.reset()
    got = lumen.to_numpy(lumen.compile(rounded)(x, w).to(dtype="float32"))
    want = lumen.to_numpy(lumen.compile(both)(x, w)[0].to(dtype="float32"))
    np.testing.assert_array_equal(got, want)
    half = lambda x, w: F.matmul(x, w, "float32", "float32").half()  # noqa: E731
    lumen.config.compiler.contraction_epilogues = False
    try:
        assert [t.split("[")[0] for t in dots(half, x, w)] == ["dot_general", "cast"]
    finally:
        lumen.config.compiler.reset()


@pytest.mark.mps
def test_matmul_epilogue_reads_the_rounded_output():
    """A fused epilogue reads the matmul's output as its dtype rounds it
    (bf16, then cast up for a silu in float32), as traced; silu of the
    float32 accumulator is a dot_general with output_dtype float32."""
    rng = np.random.default_rng(0)
    try:
        x, w = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((64, 96), (96, 80))
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    dims = (((1,), (0,)), ((), ()))
    rounded = lumen.compile(lambda x, w: _silu((x @ w).float()).to(dtype=x.dtype))(x, w)
    accum = lumen.compile(lambda x, w: _silu(prims.dot_general(x, w, dims, "float32", "float32")).to(dtype=x.dtype))(
        x, w
    )
    # As traced: the bf16 product, then silu in float32 (numpy, from the
    # unfused plan's matmul).
    h = lumen.to_numpy(lumen.compile(lambda x, w: (x @ w).float())(x, w)).astype(np.float64)
    rounded, accum = (lumen.to_numpy(t.to(dtype="float32")) for t in (rounded, accum))
    np.testing.assert_allclose(rounded, h / (1 + np.exp(-h)), rtol=1e-2, atol=1e-2)
    assert not np.array_equal(rounded, accum)


@pytest.mark.mps
def test_matmul_epilogue_takes_runtime_scalars_by_value():
    """A module's float in a matmul's epilogue is a kernel argument: a new
    value runs the same kernel."""
    rng = np.random.default_rng(0)
    w, x = rng.standard_normal((32, 48)).astype(np.float32), rng.standard_normal((16, 32)).astype(np.float32)
    weight = lumen.empty([32, 48], device="meta")
    f = lumen.compile(lambda m, a: m(a), device="mps")
    try:
        f(_ScaledMatmul(weight, 0.5), lumen.empty([16, 32], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    _ScaledMatmul(weight, 0.5).load_state_dict({"w": lumen.from_numpy(w)})
    for s in (0.5, 3.0):
        out = lumen.to_numpy(f(_ScaledMatmul(weight, s), lumen.from_numpy(x)))
        np.testing.assert_allclose(out, (x @ w) * s, rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(lambda m, a: m(a))(_ScaledMatmul(weight, 0.5), lumen.empty([16, 32], device="meta"))
    (step,) = lumen.graph.Plan(graph, "mps", parameters=[2], scalars=[1]).steps()
    assert ("s1", "float32", []) in step["inputs"] and "constant float &in" in step["fusion"]["source"]


class _Gated(lumen.nn.Module):
    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1) * (x @ self.w3)


class _Relu(lumen.nn.Module):
    w1: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1)


@pytest.mark.mps
def test_matmul_epilogue_reads_packed_weights_in_place():
    """A weight placed side by side with another (where dots merge) is a
    strided view; a matmul with its epilogue reads it in place, as the
    matmul alone does: no copy each call."""
    from lumen.profiler import ProfilerActivity, profile

    w1, w3 = lumen.empty([32, 48], device="meta"), lumen.empty([32, 48], device="meta")
    gated, relu = lumen.compile(lambda m, x: m(x), device="mps"), lumen.compile(lambda m, x: m(x), device="mps")
    try:
        gated(_Gated(w1, w3), lumen.empty([16, 32], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    relu(_Relu(w1), lumen.empty([16, 32], device="meta"))
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 48)).astype(np.float32)
    _Gated(w1, w3).load_state_dict({"w1": lumen.from_numpy(w), "w3": lumen.from_numpy(w)})
    x = rng.standard_normal((16, 32)).astype(np.float32)
    xt = lumen.from_numpy(x).to("mps")
    np.testing.assert_allclose(lumen.to_numpy(relu(_Relu(w1), xt)), np.maximum(x @ w, 0), rtol=1e-5, atol=1e-5)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        relu(_Relu(w1), xt)
    ops = [e for e in prof.events() if e["kind"] == "op"]
    assert any(e["name"].startswith("dot_general →") for e in ops)
    # (x, copied into the workspace, is made contiguous; the weight never.)
    copied = [e["inputs"] for e in ops if e["name"] == "lumen::contiguous"]
    assert [("float32", [32, 48])] not in copied, copied


def test_division_by_a_scalar_is_a_product_with_its_reciprocal():
    """``x / c`` (a Python number, or a runtime scalar) traces as
    ``x * (1 / c)``; by zero, and of tensors, it stays a division."""
    x = lumen.from_numpy(np.random.default_rng(0).standard_normal(1000).astype(np.float32))
    graph = str(lumen.make_graph(lambda x: x / 3)(x))
    assert "mul" in graph and "div" not in graph, graph
    out = lumen.to_numpy(lumen.compile(lambda x: x / 3)(x))
    np.testing.assert_array_equal(out, lumen.to_numpy(x) * np.float32(1 / 3))
    graph = str(lumen.make_graph(lambda x, s: x / s)(x, 3.0))
    assert graph.count("div") == 1 and "mul" in graph, graph
    assert "div" in str(lumen.make_graph(lambda x: x / 0.0)(x))
    assert "div" in str(lumen.make_graph(lambda x: x / x)(x))


def test_matmul_takes_accum_and_output_dtypes():
    """F.matmul takes ``accum_dtype`` and ``output_dtype`` (required); ``@``
    infers them (float32 accumulation, the inputs' dtype)."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")
    y = lumen.empty([8, 3], dtype="bfloat16", device="meta")
    assert "accum_dtype=f32 output_dtype=bf16" in str(lumen.make_graph(lambda x, y: x @ y)(x, y))
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "float32", "float32"))(x, y)
    assert "accum_dtype=f32 output_dtype=f32" in str(g)
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "bfloat16", "bfloat16"))(x, y)
    assert "accum_dtype=bf16 output_dtype=bf16" in str(g)
    with pytest.raises(TypeError, match="output_dtype"):
        lumen.make_graph(lambda x, y: F.matmul(x, y))(x, y)
    a, b = (lumen.ones(s, dtype="bfloat16") for s in ([4, 8], [8, 3]))
    assert lumen.compile(lambda x, y: F.matmul(x, y, "float32", "float32"))(a, b).dtype == "float32"


def test_rounding_a_contraction_then_widening_it_warns():
    """A matmul or attention's scores output in bf16 (accumulated in
    float32) and cast back to float32, through a scale, warns: outputting
    float32 is more accurate. Outputting float32, or not widening, does
    not."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")

    def scores(x):
        s = (x @ x.t()) * 0.5
        return s.float()

    # At the line computing the matmul, naming the cast's.
    with pytest.warns(UserWarning, match=rf"cast at {re.escape(__file__)}:\d+") as record:
        lumen.make_graph(scores)(x)
    (w,) = record
    assert (w.filename, w.lineno) == (__file__, scores.__code__.co_firstlineno + 1)
    with pytest.warns(
        UserWarning,
        match=r"dot_general\(bf16\[4,8\], bf16\[8,4\]\) -> bf16\[4,4\] accumulates in f32 but outputs bf16, then dot_general → mul → cast\(bfloat16 -> float32\)",
    ):
        lumen.make_graph(lambda x: ((x @ x.t()) * 0.5).float())(x)
    with pytest.warns(UserWarning, match="output_dtype=f32"):
        lumen.compile(lambda x: F.softmax((x @ x.t()).float(), -1))(x)
    q = lumen.empty([1, 4, 2, 8], dtype="bfloat16", device="meta")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lumen.make_graph(lambda x: F.matmul(x, x.t(), "float32", "float32") * 0.5)(x)
        lumen.make_graph(lambda x: F.relu(x @ x.t()))(x)
        lumen.make_graph(lambda q: F.flash_attention(q, q, q))(q)


class _Counter(lumen.nn.Module):
    n: lumen.Tensor
    w: lumen.Tensor


def test_copy_into_a_weight_is_written_back():
    """w.copy_(value) in a compiled function assigns a module's weight: the
    compiled function writes it back after each call (an optimizer's step),
    and the next call reads it; a call compiling on meta tensors does not."""
    counter = _Counter(lumen.empty([], device="meta"), lumen.empty([3], device="meta"))

    def step(c, x):
        c.n.copy_(c.n + 1.0)
        c.w.copy_(c.w + x * c.n)
        return c.n * 1.0

    step = lumen.compile(step, device="cpu")
    step(counter, lumen.empty([3], device="meta"))
    x = np.array([1.0, 2.0, 3.0], np.float32)
    for k in range(1, 4):
        assert float(lumen.to_numpy(step(counter, lumen.from_numpy(x)))) == k
    # w = x * (1 + 2 + 3), read back by a function returning it.
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda c: c.w * 1.0)(counter)), 6 * x)
    with pytest.raises(TypeError, match="copy_"):
        lumen.make_graph(lambda c: c.w.copy_(c.n))(counter)


class _Steps(lumen.nn.Module):
    w: lumen.Tensor
    t: float
    lrs: tuple


def test_copy_into_a_modules_float_is_written_back():
    """A module's float (a runtime scalar) assigned with copy_ is written
    back into the module, on the host, after each call: the next call
    passes it (a step count). One in a tuple cannot be."""
    steps = _Steps(lumen.empty([2], device="meta"), 0.0, (0.5,))

    def step(s):
        s.t.copy_(s.t + 1.0)
        s.w.copy_(s.w + s.t)
        return s.t * 1.0

    step = lumen.compile(step, device="cpu")
    for k in range(1, 4):
        step(steps)
        assert steps.t == k
    # w = 1 + 2 + 3.
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda s: s.w * 1.0)(steps)), [6.0, 6.0])

    def assign_tuple(s):
        s.lrs[0].copy_(s.lrs[0] * 2.0)
        return s.w * 1.0

    with pytest.raises(TypeError, match="tuple"):
        lumen.compile(assign_tuple, device="cpu")(steps)


class _ReluMLP(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        return F.relu(x @ self.w1) @ self.w2


@pytest.mark.mps
def test_matmul_epilogue_writes_values_the_backward_reads():
    """In a training step relu's backward reads x @ w1: the relu still runs
    in the matmul's kernel, which writes x @ w1 too (XLA's GELU_AUX); the
    step agrees exactly with the unfused one."""
    rng = np.random.default_rng(0)
    w1, w2 = (rng.standard_normal(s).astype(np.float32) * 0.1 for s in ((64, 96), (96, 32)))
    x = rng.standard_normal((48, 64)).astype(np.float32)

    def step(m, x):
        loss = F.sum(F.tanh(m(x)))
        loss.backward()
        return loss, [p.grad for p in m.parameters()]

    results = []
    for epilogues in (True, False):
        lumen.config.compiler.contraction_epilogues = epilogues
        try:
            model = _ReluMLP(lumen.empty([64, 96], device="meta"), lumen.empty([96, 32], device="meta"))
            f = lumen.compile(step, device="mps")
            try:
                f(model, lumen.empty([48, 64], device="meta"))
            except RuntimeError as e:
                pytest.skip(str(e))
            model.load_state_dict({"w1": lumen.from_numpy(w1), "w2": lumen.from_numpy(w2)})
            loss, grads = f(model, lumen.from_numpy(x).to("mps"))
            results.append([lumen.to_numpy(t) for t in (loss, *grads)])
            plan = lumen.graph.Plan(lumen.make_graph(step)(model, lumen.from_numpy(x)), "mps")
            labels = [s["label"] for s in plan.steps()]
            if epilogues:
                assert "dot_general → max" in labels, labels
        finally:
            lumen.config.compiler.reset()
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)


@pytest.mark.mps
def test_concatenate_runs_unfused():
    """With fusion off, a concatenate (alone a fusion on MPS) still runs."""
    try:
        a = lumen.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to("mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(lambda a: prims.concatenate([a, a * 2.0], 1))(a)
    plan = lumen.graph.Plan(graph, "mps", fuse=False)
    (out,) = plan.run([a])
    np.testing.assert_array_equal(lumen.to_numpy(out), np.concatenate([lumen.to_numpy(a), 2 * lumen.to_numpy(a)], 1))


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.mps)])
def test_cpu_inside_a_compiled_function(device):
    """``x.cpu()`` (and ``x.to("cpu")``, with a dtype too) inside a compiled
    function, as in PyTorch: a traced tensor has a device (the plan's, or
    ``cpu`` once on the host), and ``cpu()`` of a CPU tensor is the tensor
    itself (assigning it assigns the argument); another device's is copied
    to the host, its own tensor, returned as a CPU tensor."""
    devices = []

    def f(x):
        y = x * 2.0
        h = y.cpu()
        devices.extend([x.device, h.device, (h + 1.0).device, h.to(device).device])
        return h, F.sum(h + 1.0).to("cpu"), x.to("cpu", "float16"), x.to(device="cpu"), y, x.cpu().add_(0.5), x

    try:
        g = lumen.compile(f, device=device)
        first = g(lumen.tensor([1.0, 2.0]))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert devices == [device, "cpu", "cpu", device]
    h, total, half, moved, y, added, x = first
    assert all(t.device == "cpu" for t in (h, total, half, moved, added))
    assert y.device == x.device == device
    assert half.dtype == "float16"
    assert [h.tolist(), total.tolist(), half.tolist()] == [[2.0, 4.0], [8.0], [1.0, 2.0]]
    # On the CPU, ``x.to(device="cpu")`` and ``x.cpu()`` are ``x``: adding
    # to one adds to all.
    expected = [1.5, 2.5] if device == "cpu" else [1.0, 2.0]
    assert added.tolist() == [1.5, 2.5] and moved.tolist() == x.tolist() == expected


def test_python_values_inside_a_compiled_function_raise():
    """A traced tensor has no value while tracing: ``item``, ``tolist`` and
    ``float`` raise (``cpu`` keeps a tensor: it works)."""
    for body in (lambda x: x.item(), lambda x: x.tolist(), lambda x: float(F.sum(x))):
        with pytest.raises((AttributeError, TypeError)):
            lumen.compile(body)(lumen.tensor([1.0]))


@pytest.mark.mps
def test_cpu_inside_a_compiled_function_transfers():
    """``.cpu()`` inside a compiled MPS function splits it into stages: the
    ops before it on the device, then its copy to the host (``to_host``:
    ``lumen::copy_d2h``, its ``lumen::wait`` inside), then the ops reading
    it on the host, compiled for the CPU, as PyTorch runs ops on CPU
    tensors (no kernels), the result a CPU tensor. An op reading host and
    device values raises PyTorch's error. The gradient comes back to the
    device (``to_device``)."""

    def mid(x):
        return (x * 3.0).cpu() * 2.0 + 1.0

    x = lumen.tensor([1.0, 2.0])
    try:
        f = lumen.compile(mid, device="mps")
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
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
    assert y.device == "cpu" and y.tolist() == [7.0, 13.0]
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        lumen.compile(lambda x: x.cpu() + x, device="mps")(x)
    grad = lumen.compile(lumen.grad(lambda x: F.sum((x * 3.0).cpu() * 5.0).to("mps")), device="mps")(x)
    assert grad.device == "mps" and grad.tolist() == [15.0, 15.0]
