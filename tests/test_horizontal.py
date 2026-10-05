"""Horizontal loop fusion on MPS (``lumen/compiler/mps/horizontal.rs``):
independent small loop fusions of as many elements, one kernel."""

import re

import numpy as np
import pytest

import lumen
import lumen.functional as F

pytestmark = pytest.mark.mps


def rand(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


@pytest.fixture
def compiler():
    """``lumen.config.compiler``, reset to its defaults after the test."""
    yield lumen.config.compiler
    lumen.config.compiler.reset()


def _device(*arrays):
    try:
        return [lumen.from_numpy(a).to("mps") for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))


def _steps(f, *args):
    return lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()


def _merged(label):
    """Whether a step's (or kernel's) label is a horizontal fusion's: its
    members' labels, ``Nx`` before one N share, joined by ``|``."""
    return " | " in label or re.match(r"\d+x ", label) is not None


def _independent(a, b, c):
    # Three chains, none reading another's value, of as many elements (b
    # another shape), and a lone cast.
    return F.exp(a) * 2.0 + 1.0, (b - 1.0) * b, c.bfloat16()


def test_independent_loop_fusions_are_one_kernel(compiler):
    """Independent loop fusions (and a lone cast) of as many elements are
    one step, each output what it was alone, bit for bit; with
    ``horizontal_fusion`` off, a step each."""
    arrays = rand(64, 32), rand(32, 64, seed=1), rand(16, 128, seed=2)
    args = _device(*arrays)
    steps = _steps(_independent, *args)
    assert len(steps) == 1, [s["label"] for s in steps]
    assert len(steps[0]["extra_outputs"]) == 2
    got = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(_independent)(*args)]
    compiler.horizontal_fusion = False
    assert len(_steps(_independent, *args)) == 3
    alone = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(_independent)(*args)]
    for g, a in zip(got, alone):
        np.testing.assert_array_equal(g, a)
    want = lumen.compile(_independent, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, lumen.to_numpy(w.to(dtype="float32")), rtol=1e-6)


def test_fusions_of_one_chain_are_labelled_by_their_count():
    """Members computing the same primitives are labelled once, with how
    many there are, as merged dots are (``3x dot_general``)."""
    args = _device(rand(64, 32), rand(64, 32, seed=1), rand(64, 32, seed=2))
    steps = _steps(lambda *xs: tuple(F.exp(x) * 2.0 for x in xs), *args)
    assert len(steps) == 1, [s["label"] for s in steps]
    label = steps[0]["label"]
    assert label.startswith("3x ") and " | " not in label, label


def test_different_sizes_are_not_fused():
    """Fusions of different element counts stay apart (no concatenated
    buffer, XLA's other form)."""
    args = _device(rand(64, 32), rand(16, 16, seed=1))
    steps = _steps(lambda a, b: (a * 2.0 + 1.0, b * 3.0 - 1.0), *args)
    assert len(steps) == 2, [s["label"] for s in steps]


def test_dependent_fusions_are_not_fused():
    """A fusion reading another's value, or one computed from it (through a
    reduction between them), is not fused with it: that would be a cycle."""
    (a,) = _device(rand(64, 32))

    def f(a):
        x = F.exp(a) + 1.0
        return x, x * F.sum(x)

    steps = _steps(f, a)
    assert not any(_merged(s["label"]) for s in steps), [s["label"] for s in steps]
    got = lumen.compile(f)(a)
    want = lumen.compile(f, device="cpu")(lumen.from_numpy(lumen.to_numpy(a)))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-5)


def test_fusions_bind_at_most_a_kernels_buffers():
    """Many independent fusions are fused in groups, each kernel binding
    at most 30 buffers (Metal's 31, less the element count)."""
    arrays = [rand(8, 8, seed=k) for k in range(40)]
    args = _device(*arrays)

    def f(*xs):
        return tuple(x * 2.0 + float(k) for k, x in enumerate(xs))

    steps = _steps(f, *args)
    assert 2 <= len(steps) < 40, [s["label"] for s in steps]
    for s in steps:
        assert len(s["inputs"]) + 1 + len(s["extra_outputs"]) <= 30
    got = lumen.compile(f)(*args)
    for k, (g, a) in enumerate(zip(got, arrays)):
        np.testing.assert_array_equal(lumen.to_numpy(g), a * np.float32(2.0) + np.float32(k))


class _Parameters(lumen.nn.Module):
    xs: list


def test_in_place_updates_fused(compiler):
    """An optimizer's updates of its parameters, each written in place over
    the parameter, fused: each parameter as updated alone, every step, the
    same bits each run."""
    shapes = [(64, 64), (64, 64), (32, 128), (4096,)]

    def run(horizontal):
        compiler.horizontal_fusion = horizontal
        params = _Parameters([lumen.empty(list(s), "float32", device="meta") for s in shapes])

        def step(p, *grads):
            for x, g in zip(p.xs, grads):
                x.copy_(x * 0.99 - 0.1 * g)
            return grads[0] * 1.0

        step = lumen.compile(step, device="mps")
        step(params, *[lumen.empty(list(s), "float32", device="meta") for s in shapes])
        params.load_state_dict({f"xs.{k}": lumen.from_numpy(rand(*s, seed=k)) for k, s in enumerate(shapes)})
        grads = _device(*(rand(*s, seed=10 + k) for k, s in enumerate(shapes)))
        with lumen.profiler.profile(activities=[lumen.profiler.ProfilerActivity.MPS]) as prof:
            for _ in range(3):
                step(params, *grads)
            lumen.mps.synchronize()
        kernels = [e["name"] for e in prof.events() if e["kind"] == "gpu"]
        return [lumen.to_numpy(x._placed("mps")) for x in params.xs], kernels

    fused, kernels = run(True)
    assert any(_merged(k) for k in kernels), kernels
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
