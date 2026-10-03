"""Compiled execution: tracing with lumen.compile, the torch-like API on
traced tensors, and the strict primitives (lumen/graph/)."""

import dataclasses

import numpy as np
import pytest

import lumen
from lumen import prims

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
    graph = lumen.make_graph(lambda x: x.exp().sum(-1))(lumen.zeros([2, 3]))
    assert str(graph) == (
        "{ lambda %0:f32[2,3]. let\n"
        "    %1:f32[2,3] = exp %0\n"
        "    %2:f32[2] = reduce_sum[axes=(1,)] %1\n"
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
        return x if x.sum() > 0 else -x

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
    # Sums stay in the tensor's dtype (integers wrap) unless given one.
    u8 = lumen.tensor([200, 100], dtype="uint8")
    assert run(lambda x: x.sum(), u8).tolist() == 44
    assert run(lambda x: x.sum(dtype="int64"), u8).tolist() == 300
    # Explicit conversions are how dtypes change.
    assert run(lambda i: i.float().exp(), i).dtype == np.float32
    assert run(lambda i, f: i.to(f.dtype) + f, i, f).tolist() == [2.0, 4.0]
    assert run(lambda i: i.softmax(0, dtype="float32"), i).dtype == np.float32
    errors = [
        (lambda i, f: i + f, (i, f), "dtypes float32 and int64"),
        (lambda h, f: lumen.maximum(h, f), (half, lumen.tensor([0.0])), "dtypes float16 and float32"),
        (lambda f, h: lumen.where(f > 0, f, h), (lumen.tensor([1.0]), half), "dtypes float16 and float32"),
        (lambda h, f: h @ f, (lumen.zeros([2, 2], dtype="float16"), lumen.zeros([2, 2])), "dtypes float16 and float32"),
        (lambda i: i + 1.5, (i,), "int64 tensor and the float 1.5"),
        (lambda b: b + 1, (lumen.tensor([True]),), "bool tensor and the int 1"),
        (lambda i: i / 2, (i,), "true division"),
        (lambda i: i.exp(), (i,), "exp needs a floating-point tensor"),
        (lambda i: i.softmax(0), (i,), "softmax needs a floating-point tensor"),
    ]
    for fn, args, message in errors:
        with pytest.raises(TypeError, match=message):
            lumen.compile(fn)(*args)


def test_broadcasting():
    a, b = rand(4, 1, 3), rand(5, 1, seed=1)
    np.testing.assert_allclose(run(lambda a, b: a * b - a, lumen.from_numpy(a), lumen.from_numpy(b)), a * b - a, rtol=1e-6)
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


def test_reductions():
    x = rand(2, 3, 4)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: x.sum(), t), x.sum(), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: x.sum((0, -1), keepdim=True), t), x.sum((0, 2), keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: x.mean(1), t), x.mean(1), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: x.amax(-1), t), x.max(-1))
    np.testing.assert_array_equal(run(lambda x: x.max(), t), x.max())
    with pytest.raises(RuntimeError, match="floating point"):
        lumen.compile(lambda x: x.mean())(lumen.tensor([1, 2]))


def test_softmax_and_friends():
    x = rand(3, 5)
    e = np.exp(x - x.max(-1, keepdims=True))
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: x.softmax(-1), t), e / e.sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: lumen.softmax(x, dim=0).sum(0), t), np.ones(5), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: x.log_softmax(1), t), np.log(e / e.sum(-1, keepdims=True)), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(run(lambda x: x.sigmoid(), t), 1 / (1 + np.exp(-x)), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: x.relu(), t), np.maximum(x, 0))


def test_comparisons_and_where():
    x, y = rand(6), rand(6, seed=1)
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        np.testing.assert_array_equal(run(lambda a, b: getattr(a, op)(b), tx, ty), getattr(x, op)(y))
    np.testing.assert_array_equal(run(lambda a, b: lumen.where(a > b, a, 0.0), tx, ty), np.where(x > y, x, 0.0))
    np.testing.assert_array_equal(run(lumen.minimum, tx, ty), np.minimum(x, y))
    nan = lumen.tensor([float("nan"), 1.0])
    assert np.isnan(run(lumen.maximum, nan, lumen.tensor([0.0, 0.0]))[0])


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
        lumen.compile(lambda x: x.sum(3))(t)


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
        h = x + scores.softmax(-1) @ x
        return (h @ w1).relu().mean(-1), h.amax(0), lumen.where(h > 0, h, -h).sum()

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
    graph = lumen.make_graph(lambda x: (x.exp() + 1).reshape(-1).tanh())(lumen.zeros([16, 16]))
    plan = lumen.graph.Plan(graph)
    # add reads exp and the broadcast 1 and writes a third 1 KiB buffer; the
    # 0-d constant fits in exp's gap, and the reshape costs nothing.
    assert "reshape" not in str(plan)
    assert plan.workspace_bytes == 3 * 1024
    x = rand(16, 16)
    np.testing.assert_allclose(lumen.to_numpy(plan.run([lumen.from_numpy(x)])[0]), np.tanh(np.exp(x) + 1).ravel(), rtol=1e-6)
    with pytest.raises(ValueError, match="must be f32"):
        plan.run([lumen.zeros([16, 16], dtype="float64")])


def test_graph_and_plan_describe_themselves():
    graph = lumen.make_graph(lambda x: (x * 2.0).exp().sum(-1))(lumen.zeros([2, 3]))
    nodes = graph.nodes()
    assert [n["primitive"] for n in nodes] == ["full", "broadcast_in_dim", "mul", "exp", "reduce_sum"]
    assert nodes[-1]["text"] == "reduce_sum[axes=(1,)]" and nodes[-1]["fusion"] is None
    assert graph.inputs() == [0] and graph.outputs() == [nodes[-1]["output"]]
    steps = lumen.graph.Plan(graph).steps()
    assert steps[-1]["output"] == ("out0", "float32", [2])
    assert steps[2]["inputs"][0] == ("in0", "float32", [2, 3])


@pytest.mark.mps
def test_mps_plan_steps_and_profiled_kernels():
    try:
        x = lumen.ones([64, 128], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    from lumen.profiler import ProfilerActivity, profile

    graph = lumen.make_graph(lambda x: (x * 2.0 + 1.0).tanh().sum(-1))(x)
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
    kernels = [e["kernel"] for e in prof.events() if e["kind"] == "gpu"]
    assert kernels == [fusion["kernel"]]
    assert plan.run([x])[0].tolist() == pytest.approx([float(np.tanh(3.0)) * 128] * 64)


def test_dump_graph(tmp_path):
    f = lumen.compile(lambda x, w: (x @ w).relu())
    with pytest.raises(RuntimeError, match="call <lambda> first"):
        f.dump_graph(tmp_path / "none.html")
    f(lumen.zeros([2, 3]), lumen.zeros([3, 4]))
    data = f.dump_graph(tmp_path / "graph.html", json_path=tmp_path / "graph.json")
    page = (tmp_path / "graph.html").read_text()
    assert page.startswith("<!doctype html>") and "<title>&lt;lambda&gt; · lumen graph</title>" in page
    assert data["inputs"] == ["f32[2,3]", "f32[3,4]"] and data["device"] == "cpu"
    assert [n["label"] for n in data["views"]["traced"]["nodes"] if n["kind"] == "node"] == [
        "dot_general", "full", "broadcast_in_dim", "max"]
    # Another signature, traced for the dump.
    data = f.dump_graph(tmp_path / "f16.html", lumen.zeros([5, 3], dtype="float16"), lumen.zeros([3, 4], dtype="float16"))
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
        return (x @ w).relu(), x.sum(-1)

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
    np.testing.assert_allclose(lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2 + x.sum(-1, keepdims=True), rtol=1e-5, atol=1e-4)
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
    dot = lambda a, b: prims.dot_general(a, b, (((0, 2), (2, 0)), ((), ())))  # noqa: E731
    graph = lumen.make_graph(dot)(lumen.zeros([3, 2, 4]), lumen.zeros([4, 5, 3]))
    steps = [s["primitive"] for s in lumen.graph.Plan(graph, "mps", fuse=False).steps()]
    assert steps == ["transpose", "transpose", "dot_general"]
    # A split reduction's partials are scratch in the workspace.
    graph = lumen.make_graph(lambda x: x.sum(-1))(lumen.zeros([4, 50_000]))
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
    matmul, its halves read in place by the fused gate on MPS."""
    x, w13 = rand(16, 32), rand(32, 128, seed=1)
    try:
        args = [lumen.from_numpy(a).to(device) for a in (x, w13)]
    except RuntimeError as e:
        pytest.skip(str(e))

    def gated(x, w13):
        gate, up = (x @ w13).chunk(2, -1)
        return gate.relu() * up

    out = lumen.to_numpy(lumen.compile(gated)(*args))
    h = x @ w13
    np.testing.assert_allclose(out, np.maximum(h[:, :64], 0) * h[:, 64:], rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(gated)(*args)
    steps = [s["primitive"] for s in lumen.graph.Plan(graph, device).steps()]
    if device == "mps":
        assert steps[0] == "dot_general" and len(steps) == 2 and steps[1].startswith("slice -> slice")
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
        return (x @ self.w1).relu() * (x @ self.w3)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_dots_sharing_an_operand_merge(device):
    """relu(x @ w1) * (x @ w3) with weights w1 and w3: on MPS, one matmul of
    x and a block holding w1 and w3 side by side (XLA's DotMerger, with the
    weights placed together instead of concatenated)."""
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
        assert steps[0] == "dot_general" and len(steps) == 2 and "concatenate" not in steps
        # Named for the dots it computes, in the plan and the profiler.
        assert plan.steps()[0]["label"] == "2x dot_general"
        from lumen.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            f(model, lumen.from_numpy(x))
            lumen.mps.synchronize()
        names = {(e["kind"], e["name"]) for e in prof.events()}
        assert {("op", "2x dot_general"), ("gpu", "2x dot_general")} <= names
    else:
        assert plan.packed == [] and steps.count("dot_general") == 2


class Attention(lumen.nn.Module):
    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
        return (q @ k.t()).softmax(-1) @ v


@pytest.mark.mps
def test_attention_projections_merge_into_one_matmul():
    """q, k and v (x @ wq, x @ wk, x @ wv) as one matmul of x and a block
    of the three weights; the matmuls reading q and v read them in place,
    strided views of its result, and k's transpose fuses its slice."""
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
    plan = lumen.graph.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
    steps = plan.steps()
    assert plan.packed == [([1, 2, 3], 1)]
    assert [s["primitive"] for s in steps].count("dot_general") == 3
    assert [s["label"] for s in steps if s["primitive"] == "dot_general"] == ["3x dot_general", "dot_general", "dot_general"]
    assert "slice" not in [s["primitive"] for s in steps]
    views = [v for s in steps for v in s["views"] if v is not None]
    assert views == [(0, [96, 1]), (64, [96, 1])]


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
def test_softmax_is_one_primitive(device):
    """softmax over the last dimension traces to one primitive: on MPS one
    kernel (online softmax), with the scale before it fused in. Over
    another dimension it stays composite."""
    x = rand(8, 300) * 4
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def scaled(t):
        return (t * 0.5).softmax(-1)

    graph = lumen.make_graph(scaled)(t)
    assert [n["primitive"] for n in graph.nodes()][-1] == "softmax"
    s = x * 0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(scaled)(t)), e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7)
    if device == "mps":
        assert [s["label"] for s in lumen.graph.Plan(graph, "mps").steps()] == ["full -> broadcast_in_dim -> mul -> softmax"]
    e0 = np.exp(x - x.max(0, keepdims=True))
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(lambda t: t.softmax(0))(t)), e0 / e0.sum(0, keepdims=True), rtol=1e-5, atol=1e-7)
    assert "softmax" not in [n["primitive"] for n in lumen.make_graph(lambda t: t.softmax(0))(t).nodes()]


class Norm(lumen.nn.Module):
    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        return lumen.rms_norm(x, self.weight.shape, self.weight, self.eps)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_rms_norm(device):
    """lumen.rms_norm (torch.nn.functional.rms_norm), in a module too: traced
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

    f = lambda a, b: lumen.rms_norm(a + 1.0, 300, b, 1e-6)  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected(x + 1, w, 1e-6), rtol=1e-5, atol=1e-6)
    g = lambda a: lumen.rms_norm(a, [300])  # noqa: E731  (eps: float32's machine epsilon)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(g)(X)), expected(x, None, 2.0**-23), rtol=1e-5, atol=1e-6)
    assert "rms_norm" not in [n["primitive"] for n in lumen.make_graph(f)(X, W).nodes()]
    by_hand = lambda a, b: b * (a / (1e-6 + (a * a).mean(-1, keepdim=True)).sqrt())  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(by_hand)(X, W)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6)
    if device == "mps":
        steps = lambda f: [s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(X, W), "mps").steps()]  # noqa: E731
        # One kernel each: a fusion with its reduction inside (a row kernel).
        for g in (f, by_hand):
            (label,) = steps(g)
            assert "reduce_sum" in label and "sqrt" in label, label
        assert steps(f)[0].startswith("full -> broadcast_in_dim -> add -> mul -> reduce_sum")
    with pytest.raises(NotImplementedError, match="last dimension"):
        lumen.make_graph(lambda a: lumen.rms_norm(a, (8, 300)))(X)
    norm = Norm(lumen.empty([300], device="meta"), 1e-6)
    step = lumen.compile(lambda m, a: m(a), device=device)
    step(norm, lumen.empty([8, 300], device="meta"))
    norm.load_state_dict({"weight": W})
    np.testing.assert_allclose(lumen.to_numpy(step(norm, X)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", MPS])
def test_sqrt(device):
    """sqrt is a primitive (rsqrt is not: write 1 / x.sqrt()); on MPS a fused
    division by a fused sqrt is one rsqrt instruction."""
    x = rand(8, 16) ** 2 + 0.1
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    assert not hasattr(lumen, "rsqrt") and not hasattr(lumen.graph.TracedTensor, "rsqrt")
    for f, expected in [
        (lambda a: a.sqrt(), np.sqrt(x)),
        (lambda a: lumen.sqrt(a), np.sqrt(x)),
        (lambda a: 3.0 / a.sqrt(), 3 / np.sqrt(x)),
    ]:
        np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(t)), expected, rtol=1e-6)
    if device == "mps":
        for f in (lambda a: 1.0 / a.sqrt(), lambda a: a / a.sum(-1, keepdim=True).sqrt()):
            *_, last = lumen.graph.Plan(lumen.make_graph(f)(t), "mps").steps()
            assert last["label"].endswith("div") and "rsqrt(" in last["fusion"]["source"]

class ScaledNorm(lumen.nn.Module):
    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        return self.weight * (x / ((x * x).mean(-1, keepdim=True) + self.eps).sqrt())


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
