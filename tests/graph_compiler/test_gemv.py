"""Tests of matrix-vector products (a decode step's projections) on Triton: up to 8 rows times a matrix read
along K run whole on the GEMV kernel, never split over K, with the GEMM's epilogues; more rows split as before."""

import numpy as np
import pytest
from helpers import CPU, TRITON_CUDA, rand

import lumen
import lumen.functional as F


def on(device, *arrays):
    """`arrays` as bfloat16 tensors on `device`, or a skip when it is missing."""
    try:
        return [lumen.from_numpy(a).to(device).to(dtype="bfloat16") for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))


def labels(f, args, device, **options):
    """The labels of the steps `f`'s plan on `device` runs, compiled with `options` (e.g. its parameters)."""
    return [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device, **options).steps()]


def floats(t):
    """`t` on the host as float64 numpy."""
    return lumen.to_numpy(t.to("cpu").to(dtype="float32")).astype(np.float64)


@pytest.mark.parametrize("device", [CPU, TRITON_CUDA], indirect=True)
@pytest.mark.parametrize("rows", [1, 3, 8, 16])
def test_projection_with_residual(device, rows):
    """A down projection (long K) plus a residual: up to 8 rows run whole (the GEMV, its epilogue adding the
    residual), 16 split over K. The product of the bfloat16 inputs to bfloat16 rounding."""
    x, w, r = on(device, rand(rows, 8192, seed=0) / 90, rand(2048, 8192, seed=1), rand(rows, 2048, seed=2))
    f = lambda x, w, r: F.matmul(x, w.t(), "float32", "bfloat16") + r  # noqa: E731
    got = floats(lumen.compile(f)(x, w, r))
    want = floats(x) @ floats(w).T + floats(r)
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
    if device == "cuda":
        split = any("split-K" in s for s in labels(f, (x, w, r), device))
        assert split == (rows > 8), labels(f, (x, w, r), device)


@pytest.mark.parametrize("device", [CPU, TRITON_CUDA], indirect=True)
@pytest.mark.parametrize("rows", [1, 4])
def test_gated_pair(device, rows):
    """SwiGLU's gate and up projections of a few rows, merged into one product of two halves: one GEMV whose
    epilogue reads both halves (SiLU of the gate times the up). NumPy's values to bfloat16 rounding."""
    x, wg, wu = on(device, rand(rows, 2048, seed=0) / 45, rand(4096, 2048, seed=1), rand(4096, 2048, seed=2))

    def f(x, wg, wu):
        """SiLU of the gate projection times the up projection, rounded to bfloat16."""
        g = F.matmul(x, wg.t(), "float32", "float32")
        u = F.matmul(x, wu.t(), "float32", "float32")
        return (g * F.sigmoid(g) * u).to(dtype="bfloat16")

    got = floats(lumen.compile(f)(x, wg, wu))
    g, u = floats(x) @ floats(wg).T, floats(x) @ floats(wu).T
    want = g / (1 + np.exp(-g)) * u
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
    if device == "cuda":
        # The weights are parameters, packable into one block, as a model's are.
        (label,) = labels(f, (x, wg, wu), device, parameters=[1, 2], packable=[1, 2])
        assert "split-K" not in label and "logistic" in label, label
