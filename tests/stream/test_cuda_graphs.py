"""Tests of CUDA graphs: lumen.compile capturing a plan's launches and replaying them (cuda_graphs)."""

import numpy as np
import pytest
from helpers import CUDA, meta, rand

import lumen
import lumen.functional as F


@pytest.fixture
def cuda_graphs(triton_on_cuda):
    """Turns ``config.compiler.cuda.cuda_graphs`` on for the test, then back as it was."""
    before = lumen.config.compiler.cuda.cuda_graphs
    lumen.config.compiler.cuda.cuda_graphs = True
    yield
    lumen.config.compiler.cuda.cuda_graphs = before


def on_cpu(f, *arrays):
    """``f`` of ``arrays`` compiled for the CPU, as numpy."""
    return lumen.to_numpy(lumen.compile(f, device="cpu")(*[lumen.from_numpy(a) for a in arrays]))


@pytest.mark.parametrize("device", [CUDA])
def test_cuda_graphs_replay_each_call_on_its_arguments(device, cuda_graphs):
    """A plan runs as usual on its first call, is captured on its second
    and replays from then on, each call's arguments copied in: every call
    gets its own arguments' result, the CPU's. The whole-vector sum splits
    into partials allocated while capturing, which the graph keeps: tensors
    taken between replays don't land on them."""

    def f(x, y):
        """A whole-vector sum and a matmul of the first 4096 elements."""
        a, b = x[:4096].reshape(64, 64), y[:4096].reshape(64, 64)
        return F.sum(F.exp(x) * y) + F.matmul(a, b, "float32", "float32")[0, 0]

    compiled = lumen.compile(f)
    for k in range(5):
        x, y = rand(1 << 20, seed=2 * k), rand(1 << 20, seed=2 * k + 1)
        got = compiled(lumen.from_numpy(x).to(device), lumen.from_numpy(y).to(device))
        # A million float32 terms summed in another order: off by about 1e-6 of their magnitude.
        scale = np.abs(np.exp(x) * y).sum()
        np.testing.assert_allclose(lumen.to_numpy(got), on_cpu(f, x, y), rtol=0, atol=1e-5 * scale)
        lumen.full([1 << 20], float("nan"), "float32", device=device)
    (record,) = compiled._graphs.values()
    assert record is not None


class Scaled(lumen.nn.Module):
    """``tanh(x · w)``: one weight."""

    w: lumen.Tensor

    def __call__(self, x):
        """``tanh(x · w)``."""
        return F.tanh(F.matmul(x, self.w, "float32", "float32"))


@pytest.mark.parametrize("device", [CUDA])
def test_cuda_graphs_read_weights_in_place(device, cuda_graphs):
    """A captured plan reads its weights where they were placed: one
    loaded anew (in place) is read loaded, and another module's, placed
    elsewhere, makes the plan captured again."""
    x, w = rand(32, 64), rand(64, 16, seed=1)
    compiled = lumen.compile(lambda m, x: m(x), device=device)
    X = lumen.from_numpy(x).to(device)

    def model(scale):
        """A module whose weight is ``scale · w``, placed by a call."""
        m = Scaled(meta(64, 16))
        compiled(m, meta(32, 64))
        m.load_state_dict({"w": lumen.from_numpy(scale * w)})
        return m

    first = model(1)
    for _ in range(3):
        compiled(first, X)
    graph = next(iter(compiled._graphs.values()))[0]
    first.load_state_dict({"w": lumen.from_numpy(2 * w)})
    np.testing.assert_allclose(lumen.to_numpy(compiled(first, X)), np.tanh(x @ (2 * w)), rtol=1e-4, atol=1e-4)
    assert next(iter(compiled._graphs.values()))[0] is graph
    second = model(3)
    for _ in range(2):
        np.testing.assert_allclose(lumen.to_numpy(compiled(second, X)), np.tanh(x @ (3 * w)), rtol=1e-4, atol=1e-4)
    assert next(iter(compiled._graphs.values()))[0] is not graph
    np.testing.assert_allclose(lumen.to_numpy(compiled(first, X)), np.tanh(x @ (2 * w)), rtol=1e-4, atol=1e-4)


class Counted(lumen.nn.Module):
    """A weight and a step count, as an optimizer keeps them."""

    w: lumen.Tensor
    t: int


@pytest.mark.parametrize("device", [CUDA])
def test_cuda_graphs_replay_runtime_scalars_and_write_back(device, cuda_graphs):
    """A step reading a runtime scalar (the count, an int seen with two
    values), writing its weight in place and counting itself, as AdamW's
    does, is captured with the count a device input: every replay reads
    that call's count, and the weight and count are written back."""
    x, w = rand(256), rand(256, seed=1)

    def step(m, x):
        """Adds ``x · t`` to the weight, ``t`` the step's number, and returns the weight's sum."""
        m.t = m.t + 1
        m.w.copy_(m.w + x * m.t)
        return F.sum(m.w)

    compiled = lumen.compile(step, device=device)
    m = Counted(meta(256), 0)
    compiled(m, meta(256))
    m.load_state_dict({"w": lumen.from_numpy(w)})
    X = lumen.from_numpy(x).to(device)
    want = w.astype(np.float64)
    for t in range(1, 7):
        got = compiled(m, X)
        want = want + x * t
        assert m.t == t
        np.testing.assert_allclose(lumen.to_numpy(got), want.sum(), rtol=1e-5)
    np.testing.assert_allclose(lumen.to_numpy(m.w._placed(device).to("cpu")), want, rtol=1e-5)
    assert any(compiled._graphs.values())
