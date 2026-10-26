"""Tests of lumen.nn modules in compiled functions: weights placed once and shared, and scalar fields as runtime values written back."""

import dataclasses

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, meta, rand

import lumen
import lumen.functional as F


class Affine(lumen.nn.Module):
    """Affine layer `x @ w + b`."""

    w: lumen.Tensor
    b: lumen.Tensor

    def __call__(self, x):
        """Applies the layer to `x`."""
        return F.matmul(x, self.w, "float32", x.dtype) + self.b


class Stack(lumen.nn.Module):
    """Submodules in a list, a weight shared by two of them, and a static
    field."""

    layers: list
    scale: float

    def __call__(self, x):
        """Applies each layer then scales."""
        for layer in self.layers:
            x = layer(x) * self.scale
        return x


def test_modules_are_frozen_dataclasses_of_weights():
    """Modules list parameters once each, coerce static fields, freeze the rest, and validate state dicts."""
    shared = meta(4)
    m = Stack([Affine(meta(4, 4), shared), Affine(meta(4, 4), shared)], 0.5)
    assert [n for n, _ in m.named_parameters()] == ["layers.0.w", "layers.0.b", "layers.1.w"]
    assert m.parameters()[1] is shared
    m.scale = 1
    assert m.scale == 1.0 and type(m.scale) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.layers = []
    with pytest.raises(TypeError, match="float field"):
        m.scale = "1"
    with pytest.raises(ValueError, match="compile it first"):
        m.load_state_dict({n: lumen.zeros(t.shape) for n, t in m.named_parameters()})
    with pytest.raises(KeyError, match="missing"):
        m.load_state_dict({})


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_module_weights_are_placed_once_and_shared(device, triton_on_cuda):
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
    assert (y.device, y.shape) == ("meta:0", [2, 4])
    values = {n: rand(*t.shape, seed=i + 1) for i, (n, t) in enumerate(model.named_parameters())}
    model.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    x = rand(2, 4)
    h = (x @ values["layers.0.w"] + values["layers.0.b"]) * 0.5
    expected = (h @ values["layers.1.w"] + values["layers.0.b"]) * 0.5
    np.testing.assert_allclose(lumen.to_numpy(f(model, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    g = lumen.compile(lambda layer, x: layer(x), device=device)
    first = model.layers[0]
    np.testing.assert_allclose(lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2, rtol=1e-5, atol=1e-5)
    first.w.copy_(lumen.from_numpy(values["layers.0.w"] + 1))
    np.testing.assert_allclose(
        lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2 + x.sum(-1, keepdims=True), rtol=1e-5, atol=1e-4
    )
    other = Affine(meta(4, 4), meta(4))
    g(other, meta(2, 4))
    other.load_state_dict({"w": lumen.from_numpy(np.eye(4, dtype=np.float32)), "b": lumen.zeros([4])})
    np.testing.assert_allclose(lumen.to_numpy(g(other, lumen.from_numpy(x))), x, rtol=1e-6)
    with pytest.raises(TypeError, match="cannot close over a tensor"):
        lumen.compile(lambda x: F.matmul(x, first.w, "float32", x.dtype), device=device)(lumen.from_numpy(x))
    with pytest.raises(TypeError, match="whole meta tensors"):
        g(Affine(lumen.zeros([4, 4]), lumen.zeros([4])), lumen.from_numpy(x))


class ScaledNorm(lumen.nn.Module):
    """RMS norm written out by hand."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` over its last dim and scales it."""
        return self.weight * (x / F.sqrt(F.mean(x * x, -1, keepdim=True) + self.eps))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_float_fields_are_runtime_scalars(device, triton_on_cuda):
    """A module's float field is a runtime scalar: another value runs the same
    plan (no trace, no kernel compiled), read by the kernel, not baked in; an
    RMS norm with it is still one kernel on MPS."""
    x, w = rand(8, 300) * 0.3, rand(300, seed=1)
    traces = []

    def forward(m, a):
        """Counts the trace and applies `m`."""
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
    graph = lumen.make_graph(forward)(ScaledNorm(weight, 0.5), lumen.from_numpy(x))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=[2], scalars=[1])
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
        assert "constant float &in1" in step["fusion"]["source"]


class _Counter(lumen.nn.Module):
    """A step counter and a weight the step updates."""

    n: lumen.Tensor
    w: lumen.Tensor


def test_copy_into_a_weight_is_written_back():
    """w.copy_(value) in a compiled function assigns a module's weight: the
    compiled function writes it back after each call (an optimizer's step),
    and the next call reads it; a call compiling on meta tensors does not."""
    counter = _Counter(lumen.empty([], device="meta"), lumen.empty([3], device="meta"))

    def step(c, x):
        """Increments the counter and adds `x` times it to the weight."""
        c.n.copy_(c.n + 1.0)
        c.w.copy_(c.w + x * c.n)
        return c.n * 1.0

    step = lumen.compile(step, device="cpu")
    step(counter, lumen.empty([3], device="meta"))
    x = np.array([1.0, 2.0, 3.0], np.float32)
    for k in range(1, 4):
        assert float(lumen.to_numpy(step(counter, lumen.from_numpy(x)))) == k
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda c: c.w * 1.0)(counter)), 6 * x)
    with pytest.raises(TypeError, match="copy_"):
        lumen.make_graph(lambda c: c.w.copy_(c.n))(counter)


class _Steps(lumen.nn.Module):
    """A weight, a float step count and a tuple of learning rates."""

    w: lumen.Tensor
    t: float
    lrs: tuple


def test_copy_into_a_modules_float_is_written_back():
    """A module's float (a runtime scalar) assigned with copy_ is written
    back into the module, on the host, after each call: the next call
    passes it (a step count). One in a tuple cannot be."""
    steps = _Steps(lumen.empty([2], device="meta"), 0.0, (0.5,))

    def step(s):
        """Increments the step and adds it to the weight."""
        s.t.copy_(s.t + 1.0)
        s.w.copy_(s.w + s.t)
        return s.t * 1.0

    step = lumen.compile(step, device="cpu")
    for k in range(1, 4):
        step(steps)
        assert steps.t == k
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda s: s.w * 1.0)(steps)), [6.0, 6.0])

    def assign_tuple(s):
        """Assigns a float inside a tuple field, which raises."""
        s.lrs[0].copy_(s.lrs[0] * 2.0)
        return s.w * 1.0

    with pytest.raises(TypeError, match="tuple"):
        lumen.compile(assign_tuple, device="cpu")(steps)


class _Scalars(lumen.nn.Module):
    """A weight with float, int and bool fields."""

    w: lumen.Tensor
    lr: float
    step: int
    clip: bool
    heads: int


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_scalar_fields_are_assigned_as_python_values(device, triton_on_cuda):
    """A module's float, int and bool fields are assigned as Python values,
    inside a compiled function (``s.step = s.step + 1``) and outside, and
    written back as their type. A float is a runtime scalar; an int or bool
    is static until the function assigns it or it changes, then a runtime
    scalar (traced again once: no plan per value); one that cannot be (a
    condition, a shape) stays static, a plan per value."""
    traces = []

    def step(s, x):
        """Counts the trace, updates the step and learning rate, and uses every field."""
        traces.append(1)
        s.step = s.step + 1
        s.lr = s.lr * 0.5
        y = x * s.lr + s.step
        if s.clip:
            y = F.maximum(y, 0.0)
        return y.reshape([s.heads, -1])

    f = lumen.compile(step, device=device)
    s = _Scalars(lumen.empty([4], device="meta"), 1.0, 0, False, 2)
    try:
        x = lumen.from_numpy(np.full(4, -3.0, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    for k in range(1, 5):
        y = lumen.to_numpy(f(s, x))
        np.testing.assert_allclose(y, np.full((2, 2), -3.0 * 0.5**k + k), rtol=1e-6)
        assert (s.step, s.lr, s.clip) == (k, 0.5**k, False)
        assert (type(s.step), type(s.lr), type(s.clip)) == (int, float, bool)
    assert len(traces) == 2
    s.lr, s.clip = 1.0, True
    np.testing.assert_allclose(lumen.to_numpy(f(s, x)), np.full((2, 2), 3.5), rtol=1e-6)
    s.clip = False
    f(s, x)
    s.clip = True
    f(s, x)
    assert len(traces) == 4
    s.heads = 4
    assert list(f(s, x).shape) == [4, 1]
    with pytest.raises(TypeError, match="int field"):
        s.step = 1.5


class _Seen(lumen.nn.Module):
    """A weight, a float set from the device and a step counter."""

    w: lumen.Tensor
    seen: float
    step: int


def test_scalars_written_from_scalars_alone_run_on_the_host():
    """A scalar field assigned from scalars and constants alone (``step +
    1``) is computed on the host each call, from the scalars passed, never
    read back from the device (which would wait for the device's work); one
    assigned from the device's values (a loss) is read back."""
    from lumen.graph import tracer

    def step(s, x):
        """Increments the step on the host and sets `seen` from the device."""
        s.step = s.step + 1
        s.seen = F.sum(x * s.w)
        return x * 1.0

    s = _Seen(lumen.empty([3], device="meta"), 0.0, 1)
    runtime = tracer._runtime({(0, ("step",))})
    args = (s, lumen.empty([3], device="meta"))
    _, _, (_, read, _, (_, _, host)), _, _, _ = tracer._trace(step, args, "cpu", runtime, split=True)
    assert (read, host) == ([(0, ("seen",))], [(0, ("step",))])
    f = lumen.compile(step, device="cpu")
    f(s, lumen.empty([3], device="meta"))
    s.load_state_dict({"w": lumen.from_numpy(np.full(3, 2.0, np.float32))})
    for _ in range(2):
        f(s, lumen.from_numpy(np.arange(3, dtype=np.float32)))
    assert (s.seen, s.step) == (6.0, 3) and type(s.step) is int


def test_float64_runtime_scalars_are_exact():
    """A float runtime scalar of a float64 program on the CPU is the Python
    float exactly, not rounded to float32 first; float32 programs see it
    rounded once."""
    f = lumen.compile(lambda x, c: x * c, device="cpu")
    assert lumen.to_numpy(f(lumen.from_numpy(np.ones(1)), 0.1))[0] == 0.1
    assert lumen.to_numpy(f(lumen.from_numpy(np.ones(1, np.float32)), 0.1))[0] == np.float32(0.1)
