"""Tests for running fixed-weight dots on Apple's Neural Engine through Core ML."""

import json
import platform

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="the Neural Engine is Apple silicon's"
)

M, K, H = 256, 512, 1024


class MLP(lumen.nn.Module):
    """A two-layer float16 MLP: ``relu(x @ w1 + b1) @ w2``."""

    w1: lumen.Tensor
    b1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        """Returns the MLP of ``x`` in float16, its dots accumulated in float32."""
        h = F.relu(F.matmul(x, self.w1, "float32", "float16") + self.b1)
        return F.matmul(h, self.w2, "float32", "float16")


@pytest.fixture
def neural_engine():
    """Turns on the Neural Engine for one test, then resets the compiler config."""
    lumen.config.compiler.mps.neural_engine = True
    yield
    lumen.config.compiler.reset()


def _values(seed=0):
    """Returns seeded float16 MLP weights ``(w1, b1, w2)``."""
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((K, H)) / np.sqrt(K)).astype(np.float16)
    b1 = rng.standard_normal(H).astype(np.float16)
    w2 = (rng.standard_normal((H, K)) / np.sqrt(H)).astype(np.float16)
    return w1, b1, w2


def _want(x, w1, b1, w2):
    """The MLP in float32 (each dot's result rounded to float16)."""
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    h = np.maximum(f32((f32(x) @ f32(w1)).astype(np.float16)) + f32(b1), 0).astype(np.float16)
    return f32((f32(h) @ f32(w2)).astype(np.float16))


def _compiled(fn, model, x):
    """``fn`` compiled for MPS (placing ``model``'s weights), and its plan."""
    try:
        f = lumen.compile(fn, device="mps")
        f(model, lumen.empty(list(x.shape), "float16", device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(fn)(model, lumen.empty(list(x.shape), "float16", device="meta"))
    params = list(range(1, len(graph.inputs())))
    return f, lumen.graph_compiler.Plan(graph, "mps", parameters=params)


def _model():
    """Returns an MLP of meta weights with the shapes of ``_values``."""
    return MLP(*(lumen.empty(list(a.shape), "float16", device="meta") for a in _values()))


def _load(model, values):
    """Copies numpy ``values`` into ``model``'s parameters in order."""
    for t, a in zip(model.parameters(), values):
        t.copy_(lumen.from_numpy(a))


def test_neural_engine_off_by_default():
    """The flag is off: no Core ML step, every dot on lumen's kernels."""
    assert lumen.config.compiler.mps.neural_engine is False
    x = np.ones((M, K), np.float16)
    _, plan = _compiled(lambda m, x: m(x), _model(), x)
    assert not plan.neural_engine
    assert not any(s["label"].startswith("coreml") for s in plan.steps())


def test_neural_engine_runs_fixed_weights_dots(neural_engine):
    """An inference MLP (dot, bias, ReLU, dot) of fixed weights: one Core ML
    step (Core ML placing all of it on the Neural Engine, as compiling
    checked), its results float32's to float16's precision."""
    x = np.random.default_rng(1).standard_normal((M, K)).astype(np.float16)
    model = _model()
    f, plan = _compiled(lambda m, x: m(x), model, x)
    labels = [s["label"] for s in plan.steps()]
    assert plan.neural_engine and labels == [
        "coreml → dot_general → cast(float32 -> float16) → add → max → dot_general → cast(float32 -> float16)"
    ], labels
    values = _values()
    _load(model, values)
    got = lumen.to_numpy(f(model, lumen.from_numpy(x).to("mps")))
    np.testing.assert_allclose(got.astype(np.float32), _want(x, *values), rtol=2e-2, atol=2e-2)


def test_neural_engine_bakes_written_weights_in_again(neural_engine):
    """The weights are constants of the Core ML program: written
    (``copy_``), it is compiled again from their new values."""
    x = np.random.default_rng(1).standard_normal((M, K)).astype(np.float16)
    model = _model()
    f, _ = _compiled(lambda m, x: m(x), model, x)
    X = lumen.from_numpy(x).to("mps")
    for seed in (0, 1):
        values = _values(seed)
        _load(model, values)
        got = lumen.to_numpy(f(model, X)).astype(np.float32)
        np.testing.assert_allclose(got, _want(x, *values), rtol=2e-2, atol=2e-2)


def _ops(f, model, x):
    """The ops a call of ``f`` runs (its plan's steps), profiled."""
    from lumen.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU]) as p:
        lumen.to_numpy(f(model, lumen.from_numpy(x).to("mps")))
    return [e["name"] for e in p.events() if e["kind"] == "op"]


def test_neural_engine_leaves_written_weights_on_mps(neural_engine):
    """A training step's weight (written by the function) stays on lumen's
    kernels; frozen ones' dots (read, not written) run on the Neural
    Engine; small dots stay on MPS too."""

    class Frozen(lumen.nn.Module):
        """Wraps a frozen MLP and a weight the step trains."""

        mlp: MLP
        trained: lumen.Tensor

    def step(m, x):
        """Applies the MLP and the trained weight, then halves the trained weight in place."""
        y = F.matmul(m.mlp(x), m.trained, "float32", "float16")
        m.trained.copy_(m.trained * 0.5)
        return y

    x = np.ones((M, K), np.float16)
    model = Frozen(_model(), lumen.empty([K, K], "float16", device="meta"))
    f, _ = _compiled(step, model, x)
    _load(model.mlp, _values())
    model.trained.copy_(lumen.from_numpy(np.eye(K, dtype=np.float16)))
    ops = _ops(f, model, x)
    coreml = [op for op in ops if op.startswith("coreml")]
    assert coreml == [
        "coreml → dot_general → cast(float32 -> float16) → add → max → dot_general → cast(float32 -> float16)"
    ], ops
    small = MLP(*(lumen.empty(s, "float16", device="meta") for s in ([16, 32], [32], [32, 16])))
    _, plan = _compiled(lambda m, x: m(x), small, np.ones((8, 16), np.float16))
    assert not plan.neural_engine


def test_neural_engine_steps_are_profiled(neural_engine, tmp_path):
    """Each Core ML step's run is on the trace's ``Neural Engine``
    timeline."""
    from lumen.profiler import ProfilerActivity, profile

    x = np.ones((M, K), np.float16)
    model = _model()
    f, _ = _compiled(lambda m, x: m(x), model, x)
    _load(model, _values())
    X = lumen.from_numpy(x).to("mps")
    lumen.to_numpy(f(model, X))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as p:
        lumen.to_numpy(f(model, X))
    p.export_chrome_trace(str(tmp_path / "trace.json"))
    events = json.loads((tmp_path / "trace.json").read_text())["traceEvents"]
    names = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    runs = [e for e in events if e.get("cat") == "kernel" and e["name"].startswith("coreml →")]
    assert runs and all(names[e["pid"]] == "Neural Engine" for e in runs), names


class _Weights(lumen.nn.Module):
    """An MLP's float16 weights ``w``, ``b``, ``w2`` and a float32 copy of ``w``'s shape, ``w32``."""

    w: lumen.Tensor
    w32: lumen.Tensor
    b: lumen.Tensor
    w2: lumen.Tensor


# Ops after an MLP's first dot, each an extra output beside the MLP's own (or, for the weights, how the first dot
# reads them): each lowered into the Core ML program, and in it where Core ML runs them on the Neural Engine.
_TAILS = {
    "sub, mul and div": (True, lambda h, i, s: ((h - 1.0) * s / 2.0,)),
    "div": (True, lambda h, i, s: (h / (h * h + 1.0),)),
    "transpose": (True, lambda h, i, s: (h.t() * 2.0,)),
    "tanh and logistic": (True, lambda h, i, s: (F.tanh(h) + F.sigmoid(h),)),
    "exp of a negation": (True, lambda h, i, s: (F.exp(-h),)),
    "log and sqrt": (False, lambda h, i, s: (F.log(F.sigmoid(h)) + F.sqrt(F.sigmoid(h)),)),
    "select of lt": (True, lambda h, i, s: (F.where(h < 0.0, h, h * 2.0),)),
    "select of eq": (True, lambda h, i, s: (F.where(h == 0.0, h, h * 2.0),)),
    "gather of int32 indices": (True, lambda h, i, s: (prims.gather(h, i, 0),)),
    "gather of an iota": (True, lambda h, i, s: (prims.gather(h, prims.iota("int32", (M,), 0), 0),)),
    "sum": (True, lambda h, i, s: (F.sum(h, -1),)),
    "max": (True, lambda h, i, s: (F.amax(h, -1),)),
    "reshape and slice": (True, lambda h, i, s: (h.reshape(M, K, 2)[:, :8],)),
    "concatenate": (True, lambda h, i, s: (F.cat([h[:, :8], h[:, 8:16]], 1),)),
    "two outputs": (True, lambda h, i, s: (h * 2.0, h + 1.0)),
    "a transposed weight": (True, None),
    "a float32 weight": (False, None),
}


@pytest.mark.parametrize("tail", list(_TAILS))
def test_neural_engine_lowers_each_op(neural_engine, tail):
    """Each op Core ML programs run, after an MLP's first dot of fixed
    weights (``relu(x @ w + b) @ w2``) and reading it, is lowered into its
    program: one Core ML step of the MLP and the op where Core ML places
    them all on the Neural Engine (int32 indices and a float16 scalar read
    from the step's inputs; a transposed weight transposed in it), the MLP
    on lumen's kernels where it does not (log and sqrt, a float32 weight
    cast). The results lumen's MPS kernels' (the flag off) to float16's
    precision: the CPU reference of these dots is too slow to compare."""
    on_ane, ops = _TAILS[tail]

    def f(m, x, i, s):
        """The MLP of ``x`` and the tail's outputs of its first dot."""
        w = {"a transposed weight": m.w.t().t(), "a float32 weight": m.w32.to(dtype="float16")}.get(tail, m.w)
        h = F.matmul(x, w, "float32", "float16")
        out = F.matmul(F.relu(h + m.b), m.w2, "float32", "float16")
        return (out,) + (ops(h, i, s) if ops else ())

    rng = np.random.default_rng(0)
    w1, b1, w2 = _values()
    values = (w1, w1.astype(np.float32), b1, w2)
    args = ((rng.standard_normal((M, K)) / 4).astype(np.float16), rng.integers(0, M, M).astype(np.int32))
    args += (np.array(0.5, np.float16),)
    metas = [lumen.empty(list(a.shape), str(a.dtype), device="meta") for a in args]
    results = {}
    for ane in (False, True):
        lumen.config.compiler.mps.neural_engine = ane
        model = _Weights(*(lumen.empty(list(v.shape), str(v.dtype), device="meta") for v in values))
        try:
            compiled = lumen.compile(f, device="mps")
            compiled(model, *metas)
        except RuntimeError as e:
            pytest.skip(str(e))
        _load(model, values)
        out = compiled(model, *(lumen.from_numpy(a).to("mps") for a in args))
        results[ane] = [lumen.to_numpy(o.to("cpu")).astype(np.float32) for o in out]
    graph = lumen.make_graph(f)(
        _Weights(*(lumen.empty(list(v.shape), str(v.dtype), device="meta") for v in values)), *metas
    )
    plan = lumen.graph_compiler.Plan(graph, "mps", parameters=[3, 4, 5, 6])
    coreml = [s["label"] for s in plan.steps() if s["label"].startswith("coreml")]
    assert plan.neural_engine == on_ane and len(coreml) == int(on_ane), [s["label"] for s in plan.steps()]
    for got, want in zip(results[True], results[False]):
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * max(1.0, np.abs(want).max() / 64))
