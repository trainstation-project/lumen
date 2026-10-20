import numpy as np
import pytest

import lumen
import lumen.functional as F

pytestmark = pytest.mark.tpu


def floats(t):
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


@pytest.mark.parametrize("shape", [[1], [128], [129], [10, 100]])
def test_copies_round_trip_without_touching_their_neighbours(shape):
    rng = np.random.default_rng(0)
    hosts = [rng.standard_normal(shape, dtype=np.float32) for _ in range(3)]
    devices = [lumen.from_numpy(h).to("tpu") for h in hosts]
    for host, device in zip(hosts, devices):
        assert device.device == "tpu:0"
        np.testing.assert_array_equal(floats(device), host)
    bf16 = lumen.from_numpy(hosts[0]).to(dtype="bfloat16")
    np.testing.assert_array_equal(floats(bf16.to("tpu")), floats(bf16))


def check(f, lhs, rhs, dtype, tol, transposed=False):
    """``f`` compiled on TPU tensors of shapes ``lhs`` and ``rhs`` agrees
    with numpy's ``x @ w`` (``w.T`` if ``transposed``), in float32 on the
    operands the TPU got, within ``tol`` of its largest output."""
    rng = np.random.default_rng(0)
    x, w = (lumen.from_numpy(rng.standard_normal(s, dtype=np.float32)).to(dtype=dtype).to("tpu") for s in (lhs, rhs))
    y = lumen.compile(f)(x, w)
    assert y.device == "tpu:0"
    a, b = floats(x), floats(w)
    expected = a @ (b.T if transposed else b)
    assert list(y.shape) == list(expected.shape)
    assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()
    return y


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([256, 512], [512, 384]),
        ([64, 1024], [1024, 128]),
        ([1, 256], [256, 256]),
        ([2, 3, 256], [256, 128]),
        ([2, 64, 256], [2, 256, 384]),
    ],
)
def test_float32_dots_run_on_the_matrix_unit(lhs, rhs):
    y = check(lambda x, w: F.matmul(x, w, "float32", x.dtype), lhs, rhs, "float32", 1e-5)
    assert y.dtype == "float32"


def test_a_transposed_rhs():
    check(lambda x, w: F.matmul(x, w.t(), "float32", x.dtype), [64, 256], [384, 256], "float32", 1e-5, transposed=True)
    check(
        lambda x, w: F.matmul(x, w.t(), "float32", x.dtype), [8, 1024], [512, 1024], "bfloat16", 1e-2, transposed=True
    )


@pytest.mark.parametrize("dtype, tol", [("float32", 1e-5), ("bfloat16", 1e-2)])
@pytest.mark.parametrize("g, x", [([128, 256], [128, 512]), ([4096, 1024], [4096, 2048])])
def test_a_transposed_lhs(g, x, dtype, tol):
    rng = np.random.default_rng(0)
    a, b = (tpu(rng.standard_normal(s, dtype=np.float32), dtype) for s in (g, x))
    y = lumen.compile(lambda a, b: F.matmul(a.t(), b, "float32", a.dtype))(a, b)
    expected = floats(a).T @ floats(b)
    assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ([128, 512], [512, 512]),
        ([2048, 1024], [1024, 2048]),
        ([2, 16, 256], [2, 256, 256]),
    ],
)
def test_bfloat16_dots_split_their_words_by_parity(lhs, rhs):
    y = check(lambda x, w: F.matmul(x, w, "float32", x.dtype), lhs, rhs, "bfloat16", 1e-2)
    assert y.dtype == "bfloat16"


def test_a_bfloat16_dot_with_a_float32_output():
    f = lambda x, w: lumen.functional.matmul(x, w.t(), "float32", "float32")
    y = check(f, [8, 1024], [512, 1024], "bfloat16", 1e-5, transposed=True)
    assert y.dtype == "float32"


def test_small_dots_take_their_dimensions_whole():
    check(lambda x, w: F.matmul(x, w, "float32", x.dtype), [4, 8], [8, 16], "float32", 1e-5)
    check(lambda x, w: F.matmul(x, w, "float32", x.dtype), [3, 100], [100, 30], "bfloat16", 1e-2)


def test_a_dot_the_kernel_cannot_run_says_why():
    x, w = lumen.zeros([8, 128], "int32").to("tpu"), lumen.zeros([128, 128], "int32").to("tpu")
    with pytest.raises(ValueError, match="accumulating in float32"):
        lumen.compile(lambda x, w: F.matmul(x, w, "int32", "int32"))(x, w)


@pytest.mark.parametrize("shape", [[8, 128], [2, 16, 256], [1000], []])
def test_elementwise_primitives_agree_with_numpy(shape):
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal(shape, dtype=np.float32) for _ in range(2))
    x, y = (lumen.from_numpy(h).to("tpu") for h in (a, b))
    cases = [
        (lambda x, y: x + y, a + b),
        (lambda x, y: x * y, a * b),
        (lambda x, y: lumen.functional.sin(x), np.sin(a)),
        (lambda x, y: lumen.functional.exp(x), np.exp(a)),
    ]
    for f, expected in cases:
        out = lumen.compile(f)(x, y)
        assert out.device == "tpu:0"
        np.testing.assert_allclose(floats(out), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda x, y: x < y)(x, y).to("cpu")), a < b)


def test_programs_of_other_primitives_say_they_do_not_run_yet():
    """Outside a kernel, a TPU program runs dots, elementwise primitives,
    sums and Pallas kernels: another primitive (a max) says it does not run
    on a TPU yet."""
    x = lumen.ones([8, 128]).to("tpu")
    with pytest.raises(ValueError, match="reduce_max does not run on TPU yet"):
        lumen.compile(lambda a: lumen.functional.amax(a, 0), device="tpu")(x)


def test_the_profiler_times_each_tpu_program():
    """``ProfilerActivity.TPU``: each step's kernel run, timed by libtpu's
    profiler, on its device's row, inside the op that launched it: its
    kernel named and typed as an MPS kernel is."""
    from lumen.profiler import ProfilerActivity, profile

    x = lumen.ones([256, 512]).to("tpu:1")
    f = lumen.compile(lambda x: lumen.functional.exp(x) * x)
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.TPU], record_shapes=True) as prof:
        f(x)
    events = prof.events()
    runs = [e for e in events if e["kind"] == "gpu"]
    assert [e["name"] for e in runs] == ["exp", "mul"]
    assert [e["kernel"] for e in runs] == ["exp_f32_256x512", "mul_f32_256x512"]
    ops = {e["id"]: e for e in events if e["kind"] == "op"}
    for run in runs:
        assert run["device"] == "tpu:1"
        assert run["outputs"] == [("float32", [256, 512])]
        assert run["duration_us"] > 0
        assert ops[run["parent"]]["name"] == run["name"]
    assert runs[0]["end_us"] <= runs[1]["start_us"]
    (row,) = [r for r in prof.key_averages() if r.key == "exp"]
    assert row.count == 1 and row.self_device_time_total == pytest.approx(runs[0]["duration_us"], rel=1e-3)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_dot_of_two_vectors_runs_as_a_product_and_a_sum(dtype):
    """``rowsum(dO * O)`` as attention's backward traces it, a dot with no
    free dimensions (no matmul runs it): dot strength reduction's product
    and row sum, in float32."""
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal([2, 8, 1024, 128], dtype=np.float32) for _ in range(2))
    x, y = (lumen.from_numpy(t).to(dtype=dtype).to("tpu") for t in (a, b))
    out = lumen.compile(
        lambda x, y: lumen.functional.einsum("bhsd,bhsd->bhs", x, y, accum_dtype="float32", output_dtype=x.dtype)
    )(x, y)
    assert out.device == "tpu:0" and list(out.shape) == [2, 8, 1024]
    expected = (floats(x) * floats(y)).sum(-1)
    assert np.abs(floats(out) - expected).max() <= {"float32": 1e-5, "bfloat16": 1e-2}[dtype] * np.abs(expected).max()


@pytest.mark.parametrize("shape", [[64, 128], [2048, 2048], [3, 16, 256, 128], [8, 1024, 1000]])
def test_sums_over_the_last_dimension(shape):
    """Two dimensions or more, the summed one in blocks of an accumulator
    (2048: four), bfloat16 summed in float32."""
    rng = np.random.default_rng(0)
    x = lumen.from_numpy(rng.standard_normal(shape, dtype=np.float32)).to(dtype="bfloat16").to("tpu")
    out = lumen.compile(lambda x: lumen.functional.sum(x.float(), -1))(x)
    assert out.dtype == "float32" and list(out.shape) == shape[:-1]
    np.testing.assert_allclose(floats(out), floats(x).sum(-1), rtol=1e-5, atol=1e-4)


def test_a_sum_over_another_dimension_says_so():
    x = lumen.ones([64, 128]).to("tpu")
    with pytest.raises(ValueError, match="over the last dimension"):
        lumen.compile(lambda x: lumen.functional.sum(x, 0))(x)


class Projections(lumen.nn.Module):
    ws: list

    def __call__(self, x):
        return [-F.matmul(x, w.t(), "float32", x.dtype) for w in self.ws]


@pytest.mark.parametrize("dtype, tol", [("float32", 1e-5), ("bfloat16", 1e-2)], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("sizes", [[256, 128, 128], [512, 512], [1024, 2048]], ids=["qkv", "pair", "wide"])
def test_dots_of_one_input_merge_into_one_matmul(sizes, dtype, tol):
    d, m = 256, 64
    meta = lambda *shape: lumen.empty(list(shape), dtype, "meta")
    make = lambda: Projections([meta(h, d) for h in sizes])
    graph = lumen.make_graph(lambda model, x: model(x))(make(), meta(m, d))
    params = list(range(1, len(sizes) + 1))
    plan = lumen.graph_compiler.Plan(graph, "tpu", parameters=params, packable=params)
    assert plan.packed == [(params, 0)]
    labels = [s["label"] for s in plan.steps()]
    rounded = "" if dtype == "float32" else " → cast(float32 -> bfloat16)"
    assert labels == [f"{len(sizes)}x dot_general" + rounded + " → neg" * len(sizes)], labels
    rng = np.random.default_rng(0)
    values = [lumen.from_numpy(rng.standard_normal([h, d], dtype=np.float32)).to(dtype=dtype) for h in sizes]
    model = make()
    f = lumen.compile(lambda model, x: model(x), device="tpu")
    f(model, meta(m, d))
    model.load_state_dict({f"ws.{k}": v for k, v in enumerate(values)})
    x = lumen.from_numpy(rng.standard_normal([m, d], dtype=np.float32)).to(dtype=dtype).to("tpu")
    ys = f(model, x)
    for y, w in zip(ys, values):
        expected = -(floats(x) @ floats(w).T)
        assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()


def tpu(a, dtype="float32"):
    return lumen.from_numpy(a).to(dtype=dtype).to("tpu")


@pytest.mark.parametrize(
    "f, expected, labels",
    [
        (
            lambda x, w, r: lumen.functional.relu(F.matmul(x, w, "float32", x.dtype)),
            lambda x, w, r: np.maximum(x @ w, 0),
            ["dot_general → max"],
        ),
        (
            lambda x, w, r: (F.matmul(x, w, "float32", x.dtype) * 2.0 + r).to(dtype="bfloat16"),
            lambda x, w, r: (x @ w) * 2 + r,
            ["dot_general → mul → add → cast(float32 -> bfloat16)"],
        ),
        (
            lambda x, w, r: (
                F.matmul(x, w, "float32", x.dtype),
                lumen.functional.exp(F.matmul(x, w, "float32", x.dtype)) - r,
            ),
            lambda x, w, r: (x @ w, np.exp(x @ w) - r),
            ["dot_general → exp → sub"],
        ),
    ],
    ids=["relu", "residual and cast", "the dot's value read too"],
)
def test_a_dots_elementwise_consumers_are_its_epilogue(f, expected, labels):
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal([256, 512], dtype=np.float32), rng.standard_normal([512, 384], dtype=np.float32) / 16
    c = rng.standard_normal([256, 384], dtype=np.float32)
    meta = lambda *shape: lumen.empty(list(shape), "float32", "meta")
    graph = lumen.make_graph(f)(meta(256, 512), meta(512, 384), meta(256, 384))
    plan = lumen.graph_compiler.Plan(graph, "tpu")
    assert [s["label"] for s in plan.steps() if not s["label"].startswith(("full", "broadcast"))] == labels
    got = lumen.compile(f)(tpu(a), tpu(b), tpu(c))
    want = expected(a, b, c)
    for g, e in zip(got if isinstance(got, tuple) else [got], want if isinstance(want, tuple) else [want]):
        np.testing.assert_allclose(floats(g), e, rtol=1e-2, atol=1e-2 * np.abs(e).max())


class GatedPair(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        g, u = F.matmul(x, self.w1.t(), "float32", x.dtype), F.matmul(x, self.w2.t(), "float32", x.dtype)
        return g * lumen.functional.sigmoid(g) * u


class Product(lumen.nn.Module):
    ws: list

    def __call__(self, x):
        q, k, v = (F.matmul(x, w.t(), "float32", x.dtype) for w in self.ws)
        return q * k * v


@pytest.mark.parametrize("dtype, tol", [("float32", 1e-4), ("bfloat16", 2e-2)], ids=["float32", "bfloat16"])
@pytest.mark.parametrize(
    "make, steps, expected",
    [
        (
            lambda meta, h, d: GatedPair(meta(h, d), meta(h, d)),
            ["2x dot_general → logistic → mul → mul"],
            lambda x, ws: (lambda g, u: g / (1 + np.exp(-g)) * u)(x @ ws[0].T, x @ ws[1].T),
        ),
        (
            lambda meta, h, d: Product([meta(h, d) for _ in range(3)]),
            ["3x dot_general → mul → mul"],
            lambda x, ws: (x @ ws[0].T) * (x @ ws[1].T) * (x @ ws[2].T),
        ),
    ],
    ids=["swiglu", "three"],
)
def test_merged_members_combined_are_one_epilogue(make, steps, expected, dtype, tol):
    d, h, m = 256, 512, 64
    meta = lambda *shape: lumen.empty(list(shape), dtype, "meta")
    model = make(meta, h, d)
    n = len(list(model.named_parameters()))
    graph = lumen.make_graph(lambda model, x: model(x))(model, meta(m, d))
    params = list(range(1, n + 1))
    plan = lumen.graph_compiler.Plan(graph, "tpu", parameters=params, packable=params)
    assert plan.packed == [(params, 0)]
    rounded = "" if dtype == "float32" else " → cast(float32 -> bfloat16)"
    assert [s["label"] for s in plan.steps()] == [
        label.replace("dot_general", "dot_general" + rounded) for label in steps
    ]
    rng = np.random.default_rng(0)
    ws = [rng.standard_normal([h, d], dtype=np.float32) / 16 for _ in range(n)]
    x = rng.standard_normal([m, d], dtype=np.float32)
    f = lumen.compile(lambda model, x: model(x), device="tpu")
    f(model, meta(m, d))
    names = [name for name, _ in model.named_parameters()]
    model.load_state_dict({name: lumen.from_numpy(w).to(dtype=dtype) for name, w in zip(names, ws)})
    xd = tpu(x, dtype)
    wd = [floats(lumen.from_numpy(w).to(dtype=dtype)) for w in ws]
    want = expected(floats(xd), wd)
    np.testing.assert_allclose(floats(f(model, xd)), want, rtol=tol, atol=tol * np.abs(want).max())


class QKV(lumen.nn.Module):
    ws: list

    def __call__(self, x):
        return [F.matmul(x, w.t(), "float32", x.dtype) for w in self.ws]


def test_projections_backward_merges_into_two_dots():
    d, h, m = 256, 256, 128
    meta = lambda *shape: lumen.empty(list(shape), "float32", "meta")

    def step(model, x, g):
        q, k, v = model(x)
        (q * k * v).backward(g)
        return [w.grad for w in model.ws] + [x.grad]

    graph = lumen.make_graph(step)(QKV([meta(h, d) for _ in range(3)]), meta(m, d), meta(m, h))
    plan = lumen.graph_compiler.Plan(graph, "tpu", parameters=[2, 3, 4], packable=[2, 3, 4])
    assert plan.packed == [([2, 3, 4], 0)]
    labels = [s["label"] for s in plan.steps()]
    [label for label in labels if "dot_general" in label]
    assert labels == ["3x dot_general" + " → mul" * 5, "3x dot_general", "dot_general"], labels
    assert not any("concatenate" in label for label in labels), labels
    rng = np.random.default_rng(0)
    ws = [rng.standard_normal([h, d], dtype=np.float32) / 16 for _ in range(3)]
    x, g = rng.standard_normal([m, d], dtype=np.float32), rng.standard_normal([m, h], dtype=np.float32)
    model = QKV([meta(h, d) for _ in range(3)])
    f = lumen.compile(step, device="tpu")
    f(model, meta(m, d), meta(m, h))
    model.load_state_dict({f"ws.{i}": lumen.from_numpy(w) for i, w in enumerate(ws)})
    got = [floats(t) for t in f(model, tpu(x), tpu(g))]
    q, k, v = (x @ w.T for w in ws)
    dq, dk, dv = g * k * v, g * q * v, g * q * k
    expected = [dq.T @ x, dk.T @ x, dv.T @ x, dq @ ws[0] + dk @ ws[1] + dv @ ws[2]]
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())


class SwiGLU(lumen.nn.Module):
    w1: lumen.Tensor
    w3: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        g, u = F.matmul(x, self.w1.t(), "float32", x.dtype), F.matmul(x, self.w3.t(), "float32", x.dtype)
        return F.matmul(g * lumen.functional.sigmoid(g) * u, self.w2.t(), "float32", x.dtype)


def test_a_swiglu_backward_merges_its_gate_and_up():
    d, h, m = 256, 512, 128
    meta = lambda *shape: lumen.empty(list(shape), "float32", "meta")

    def step(model, x, dy):
        model(x).backward(dy)
        return [model.w1.grad, model.w3.grad, model.w2.grad, x.grad]

    make = lambda: SwiGLU(meta(h, d), meta(h, d), meta(d, h))
    graph = lumen.make_graph(step)(make(), meta(m, d), meta(m, d))
    plan = lumen.graph_compiler.Plan(graph, "tpu", parameters=[2, 3, 4], packable=[2, 3, 4])
    assert plan.packed == [([2, 3], 0)]
    labels = [s["label"] for s in plan.steps()]
    assert len(labels) == 5 and labels[0].startswith("2x dot_general →") and "2x dot_general" in labels, labels
    assert not any("concatenate" in label for label in labels), labels
    rng = np.random.default_rng(0)
    w1, w3, w2 = (rng.standard_normal(s, dtype=np.float32) / 16 for s in ([h, d], [h, d], [d, h]))
    x, dy = (rng.standard_normal([m, d], dtype=np.float32) for _ in range(2))
    model = make()
    f = lumen.compile(step, device="tpu")
    f(model, meta(m, d), meta(m, d))
    model.load_state_dict({"w1": lumen.from_numpy(w1), "w3": lumen.from_numpy(w3), "w2": lumen.from_numpy(w2)})
    got = [floats(t) for t in f(model, tpu(x), tpu(dy))]
    g, u = x @ w1.T, x @ w3.T
    s = 1 / (1 + np.exp(-g))
    dh = dy @ w2
    dg, du = dh * u * (s + g * s * (1 - s)), dh * g * s
    for a, b in zip(got, [dg.T @ x, du.T @ x, dy.T @ (g * s * u), dg @ w1 + du @ w3]):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())
