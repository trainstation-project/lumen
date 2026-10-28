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
