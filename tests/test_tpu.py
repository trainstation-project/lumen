"""Compiled programs on a TPU: their kernels lumen's own Mosaic kernels."""

import numpy as np
import pytest

import lumen

pytestmark = pytest.mark.tpu


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("transposed", [False, True])
def test_a_compiled_matmul_runs_on_the_tpu(dtype, transposed):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 512), dtype=np.float32)
    b = rng.standard_normal((384, 512) if transposed else (512, 384), dtype=np.float32)
    x, w = (lumen.from_numpy(v).to(dtype=dtype).to("tpu") for v in (a, b))
    f = lumen.compile(lambda x, w: x @ (w.t() if transposed else w))
    y = f(x, w)
    assert (y.device, y.dtype, y.shape) == ("tpu:0", dtype, [256, 384])
    # The operands as the TPU got them; the product in float32.
    a, b = (lumen.to_numpy(v.to("cpu").to(dtype="float32")) for v in (x, w))
    expected = a @ (b.T if transposed else b)
    got = lumen.to_numpy(y.to("cpu").to(dtype="float32"))
    tol = 1e-5 if dtype == "float32" else 1e-2
    assert np.abs(got - expected).max() <= tol * np.abs(expected).max()
