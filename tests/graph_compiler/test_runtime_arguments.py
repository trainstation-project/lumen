"""Tests of a compiled program run again with new runtime arguments: float scalars and tensors at other
addresses, each run computing with its own values (on CUDA, a cached Triton kernel launched from Rust)."""

import numpy as np
import pytest
from helpers import CUDA, MPS

import lumen


def _device(device):
    """Skips the test when `device` can't hold tensors."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_float_scalars_change_between_runs(device, dtype, triton_on_cuda):
    """``x + c`` for a Python float `c` is traced once and runs with each new
    `c` (zero, negatives, a whole number, tiny and huge ones), each run adding
    its own; ints are part of the program, each value traced on its own."""
    _device(device)
    x = np.arange(1000).astype(dtype)
    traces = []

    def f(x, c):
        """Counts the trace and adds `c`."""
        traces.append(1)
        return x + c

    g = lumen.compile(f, device=device)
    xt = lumen.from_numpy(x).to(device)
    for c in (4.0, 5.5, 0.0, -0.25, 16.0, 1e-30, 6e4):
        got = lumen.to_numpy(g(xt, c).to("cpu"))
        np.testing.assert_array_equal(got, x + np.asarray(c, np.float32).astype(dtype))
    assert len(traces) == 1
    for c in (4, 5, 16, 1):
        np.testing.assert_array_equal(lumen.to_numpy(g(xt, c).to("cpu")), x + np.asarray(c, dtype))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "float16", "int8"])
def test_tensors_at_other_addresses(device, dtype, triton_on_cuda):
    """One compiled ``x * 3 + 1`` on views of a buffer at each offset from 0
    to 17 elements (16-byte aligned or not, which kernels may be specialized
    on), and on fresh buffers: each run reads its own elements."""
    _device(device)
    base = (np.arange(1041) % 100).astype(dtype)
    bt = lumen.from_numpy(base).to(device)
    g = lumen.compile(lambda x: x * 3 + 1, device=device)
    for k in range(18):
        got = lumen.to_numpy(g(bt.narrow(0, k, 1024)).to("cpu"))
        np.testing.assert_array_equal(got, base[k : k + 1024] * 3 + 1)
    for seed in range(3):
        x = (np.random.default_rng(seed).integers(0, 50, 1024)).astype(dtype)
        np.testing.assert_array_equal(lumen.to_numpy(g(lumen.from_numpy(x).to(device)).to("cpu")), x * 3 + 1)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
@pytest.mark.parametrize("kernel", ["reduction", "matmul epilogue", "row"])
def test_float_scalars_in_reductions_and_matmuls(device, kernel, triton_on_cuda):
    """A Python float read inside a reduction's prologue, a matmul's
    epilogue or a softmax row is traced once and passed to each run's kernel
    as an argument: each run computes with its own value, as the CPU does."""
    _device(device)
    rng = np.random.default_rng(0)
    x, w = rng.standard_normal((64, 512)).astype(np.float32), rng.standard_normal((512, 256)).astype(np.float32)
    traces = []
    import lumen.functional as F

    def f(x, w, c):
        """Counts the trace and computes the case's kernel reading `c`."""
        traces.append(1)
        if kernel == "reduction":
            return F.sum(x * c, -1)
        if kernel == "row":
            return F.softmax(x * c, -1)
        return F.matmul(x, w, "float32", "float32") * c

    g, cpu = lumen.compile(f, device=device), lumen.compile(f, device="cpu")
    args = (lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device))
    for c in (0.5, -2.0, 3.0):
        got = lumen.to_numpy(g(*args, c).to("cpu"))
        want = lumen.to_numpy(cpu(lumen.from_numpy(x), lumen.from_numpy(w), c))
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
    assert len(traces) == 2


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_a_reduction_of_many_inputs(device, triton_on_cuda):
    """A sum of sixteen inputs' sum, one reduction fusion reading them all
    (its generated kernel's argument list wrapped one to a line): the CPU's."""
    _device(device)
    arrays = [np.random.default_rng(k).standard_normal((32, 64)).astype(np.float32) for k in range(16)]
    import lumen.functional as F

    def f(*xs):
        """The row sums of the inputs' sum."""
        total = xs[0]
        for x in xs[1:]:
            total = total + x
        return F.sum(total, -1)

    got = lumen.to_numpy(lumen.compile(f)(*(lumen.from_numpy(a).to(device) for a in arrays)).to("cpu"))
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(*(lumen.from_numpy(a) for a in arrays)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
