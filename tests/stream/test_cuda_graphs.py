"""Tests of CUDA graphs: lumen.compile capturing a plan's launches and replaying them (cuda_graphs)."""

import numpy as np
import pytest
from helpers import CUDA, meta, rand

import lumen
import lumen.functional as F


@pytest.fixture
def cuda_graphs(triton_on_cuda):
    """Turns ``config.compiler.cuda.cuda_graphs`` on for the test."""
    lumen.config.compiler.cuda.cuda_graphs = True
    yield
    lumen.config.compiler.cuda.cuda_graphs = False


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


@pytest.mark.parametrize("device", [CUDA])
def test_cuda_graphs_leave_runtime_scalars_uncaptured(device, cuda_graphs):
    """A plan with a runtime scalar (an int seen with two values) reads it
    on the host each call, so no graph captures it: it runs as usual."""
    x = rand(1024)
    compiled = lumen.compile(lambda x, n: x * n)
    X = lumen.from_numpy(x).to(device)
    for n in range(1, 5):
        np.testing.assert_allclose(lumen.to_numpy(compiled(X, n)), x * n, rtol=1e-6)
    assert not any(compiled._graphs.values())
