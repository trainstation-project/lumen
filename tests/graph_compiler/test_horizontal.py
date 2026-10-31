"""Tests of horizontal fusion on MPS and on CUDA through Triton: independent kernels merged into one launch."""

import re

import numpy as np
import pytest
from helpers import CUDA, MPS, rand

import lumen
import lumen.functional as F

DEVICES = [MPS, CUDA]

# Most buffers one horizontally fused kernel binds on each device.
MAX_BUFFERS = {"mps": 30, "cuda": 64}


@pytest.fixture
def compiler(device, triton_on_cuda):
    """``device``'s compiler settings (CUDA's running Triton), reset to their defaults after the test."""
    yield getattr(lumen.config.compiler, device)
    lumen.config.compiler.reset()


def _device(device, *arrays):
    """`arrays` on `device`; skips the test when it is unavailable."""
    try:
        return [lumen.from_numpy(a).to(device) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))


def _steps(device, f, *args):
    """The plan steps of `f` traced on `args`, compiled for `device`."""
    return lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()


def _merged(label):
    """Whether a step's (or kernel's) label is a horizontal fusion's: its
    members' labels, ``Nx`` before one N share, joined by ``|``."""
    return " | " in label or re.match(r"\d+x ", label) is not None


def _independent(a, b, c):
    """Three independent elementwise computations of equal element count, one a lone cast."""
    return F.exp(a) * 2.0 + 1.0, (b - 1.0) * b, c.bfloat16()


@pytest.mark.parametrize("device", DEVICES)
def test_independent_elementwise_fusions_are_one_kernel(device, compiler):
    """Independent elementwise fusions (and a lone cast) of as many elements are
    one step, each output what it was alone, bit for bit; with
    ``horizontal_fusion`` off, a step each."""
    arrays = rand(64, 32), rand(32, 64, seed=1), rand(16, 128, seed=2)
    args = _device(device, *arrays)
    steps = _steps(device, _independent, *args)
    assert len(steps) == 1, [s["label"] for s in steps]
    assert len(steps[0]["extra_outputs"]) == 2
    got = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(_independent)(*args)]
    compiler.horizontal_fusion = False
    assert len(_steps(device, _independent, *args)) == 3
    alone = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(_independent)(*args)]
    for g, a in zip(got, alone):
        np.testing.assert_array_equal(g, a)
    want = lumen.compile(_independent, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, lumen.to_numpy(w.to(dtype="float32")), rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_fusions_of_one_chain_are_labelled_by_their_count(device, compiler):
    """Members computing the same primitives are labelled once, with how
    many there are, as merged dots are (``3x dot_general``)."""
    args = _device(device, rand(64, 32), rand(64, 32, seed=1), rand(64, 32, seed=2))
    steps = _steps(device, lambda *xs: tuple(F.exp(x) * 2.0 for x in xs), *args)
    assert len(steps) == 1, [s["label"] for s in steps]
    label = steps[0]["label"]
    assert label.startswith("3x ") and " | " not in label, label


@pytest.mark.parametrize("device", DEVICES)
def test_fusions_of_different_record_function_calls_are_not_fused(device, compiler):
    """Fusions traced in different ``record_function`` calls (two layers'
    weight casts) stay apart, so each kernel runs in its own call's range
    when profiled; the same fusions outside any range are one."""
    args = _device(device, rand(64, 32), rand(64, 32, seed=1))

    def in_calls(a, b):
        """Each input's computation inside its own `record_function("layer")` range."""
        outs = []
        for x in (a, b):
            with lumen.profiler.record_function("layer"):
                outs.append(F.exp(x) * 2.0)
        return tuple(outs)

    steps = _steps(device, in_calls, *args)
    assert len(steps) == 2 and not any(_merged(s["label"]) for s in steps), [s["label"] for s in steps]
    assert all(s["scope"] == ["layer"] for s in steps), [s["scope"] for s in steps]
    assert len(_steps(device, lambda a, b: (F.exp(a) * 2.0, F.exp(b) * 2.0), *args)) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_different_sizes_split_the_grid(device, compiler):
    """Fusions of different element counts are one kernel too: each count
    runs on its own range of threads, each output what it was alone, bit for
    bit, under every elementwise tuning."""

    def f(a, b, c, d):
        """Two computations of 2048 elements, one of 256 and one of 7."""
        return a * 2.0 + 1.0, F.exp(b), c * 3.0 - 1.0, d * d

    args = _device(device, rand(64, 32), rand(32, 64, seed=1), rand(16, 16, seed=2), rand(7, seed=3))
    steps = _steps(device, f, *args)
    assert len(steps) == 1 and len(steps[0]["extra_outputs"]) == 3, [s["label"] for s in steps]
    got = [lumen.to_numpy(v) for v in lumen.compile(f)(*args)]
    compiler.autotune = False
    untuned = [lumen.to_numpy(v) for v in lumen.compile(f)(*args)]
    compiler.horizontal_fusion = False
    assert len(_steps(device, f, *args)) == 4
    alone = [lumen.to_numpy(v) for v in lumen.compile(f)(*args)]
    for g, u, a in zip(got, untuned, alone):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_array_equal(u, a)


@pytest.mark.parametrize("device", DEVICES)
def test_dependent_fusions_are_not_fused(device, compiler):
    """A fusion reading another's value, or one computed from it (through a
    reduction between them), is not fused with it: that would be a cycle."""
    (a,) = _device(device, rand(64, 32))

    def f(a):
        """`x` and a value computed from `x` through a reduction."""
        x = F.exp(a) + 1.0
        return x, x * F.sum(x)

    steps = _steps(device, f, a)
    assert not any(_merged(s["label"]) for s in steps), [s["label"] for s in steps]
    got = lumen.compile(f)(a)
    want = lumen.compile(f, device="cpu")(lumen.from_numpy(lumen.to_numpy(a)))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_fusions_bind_at_most_a_kernels_buffers(device, compiler):
    """Many independent fusions are fused in groups, each kernel binding
    at most the device's buffers: 30 on MPS (Metal's 31, less the element
    count), 64 on Triton."""
    arrays = [rand(8, 8, seed=k) for k in range(40)]
    args = _device(device, *arrays)

    def f(*xs):
        """One independent elementwise fusion per input."""
        return tuple(x * 2.0 + float(k) for k, x in enumerate(xs))

    steps = _steps(device, f, *args)
    assert 2 <= len(steps) < 40, [s["label"] for s in steps]
    for s in steps:
        assert len(s["inputs"]) + 1 + len(s["extra_outputs"]) <= MAX_BUFFERS[device]
    got = lumen.compile(f)(*args)
    for k, (g, a) in enumerate(zip(got, arrays)):
        np.testing.assert_array_equal(lumen.to_numpy(g), a * np.float32(2.0) + np.float32(k))


class _Parameters(lumen.nn.Module):
    """A module holding a list of parameter tensors."""

    xs: list


@pytest.mark.parametrize("device", DEVICES)
def test_in_place_updates_fused(device, compiler):
    """An optimizer's updates of its parameters, each written in place over
    the parameter, fused into one step: each parameter as updated alone,
    every step, the same bits each run."""
    shapes = [(64, 64), (64, 64), (32, 128), (4096,)]

    def run(horizontal):
        """Compiles an in-place update step with `horizontal` fusion on or off, runs it three times on
        fresh parameters and returns the parameters and the plan's step labels."""
        compiler.horizontal_fusion = horizontal
        params = _Parameters([lumen.empty(list(s), "float32", device="meta") for s in shapes])

        def step(p, *grads):
            """Updates every parameter in place from its gradient."""
            for x, g in zip(p.xs, grads):
                x.copy_(x * 0.99 - 0.1 * g)
            return grads[0] * 1.0

        metas = [lumen.empty(list(s), "float32", device="meta") for s in shapes]
        labels = [s["label"] for s in _steps(device, step, params, *metas)]
        step = lumen.compile(step, device=device)
        step(params, *metas)
        params.load_state_dict({f"xs.{k}": lumen.from_numpy(rand(*s, seed=k)) for k, s in enumerate(shapes)})
        grads = _device(device, *(rand(*s, seed=10 + k) for k, s in enumerate(shapes)))
        for _ in range(3):
            step(params, *grads)
        return [lumen.to_numpy(x._placed(device)) for x in params.xs], labels

    fused, labels = run(True)
    assert any(_merged(label) for label in labels), labels
    again, _ = run(True)
    alone, _ = run(False)
    want = [rand(*s, seed=k) for k, s in enumerate(shapes)]
    for _ in range(3):
        want = [
            x * np.float32(0.99) - np.float32(0.1) * rand(*s, seed=10 + k)
            for k, (x, s) in enumerate(zip(want, shapes))
        ]
    for f, a, b, w in zip(fused, again, alone, want):
        np.testing.assert_array_equal(f, a)
        np.testing.assert_array_equal(f, b)
        np.testing.assert_allclose(f, w, rtol=1e-5, atol=1e-6)
