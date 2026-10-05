"""``lumen.config.compiler.neural_engine``: an MPS plan runs the large
float16 dots of the weights its function does not write (and the float16
work after them) on the Apple Neural Engine, as Core ML steps
(``lumen/compiler/mps/ane``): results as computed in float32 (to the Neural
Engine's precision); none with the flag off, nor for a weight the function
writes (a training step's); the weights baked in again once written; the
steps on the profile's own timeline."""

import json
import platform

import numpy as np
import pytest

import lumen
import lumen.functional as F

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64", reason="the Neural Engine is Apple silicon's"
)

M, K, H = 256, 512, 1024


class MLP(lumen.nn.Module):
    w1: lumen.Tensor
    b1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        h = F.relu(F.matmul(x, self.w1, "float32", "float16") + self.b1)
        return F.matmul(h, self.w2, "float32", "float16")


@pytest.fixture
def neural_engine():
    lumen.config.compiler.neural_engine = True
    yield
    lumen.config.compiler.reset()


def _values(seed=0):
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
    return f, lumen.graph.Plan(graph, "mps", parameters=params)


def _model():
    return MLP(*(lumen.empty(list(a.shape), "float16", device="meta") for a in _values()))


def _load(model, values):
    for t, a in zip(model.parameters(), values):
        t.copy_(lumen.from_numpy(a))


def test_neural_engine_off_by_default():
    """The flag is off: no Core ML step, every dot on lumen's kernels."""
    assert lumen.config.compiler.neural_engine is False
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
    assert plan.neural_engine and labels == ["coreml → dot_general → add → max → dot_general"], labels
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
        mlp: MLP
        trained: lumen.Tensor

    def step(m, x):
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
    assert coreml == ["coreml → dot_general → add → max → dot_general"], ops
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
