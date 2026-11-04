"""Tests for ``preserve_precision``: a value's round trip through another float dtype dropped where the
forward reads it, the narrower dtype kept for the gradients' reads."""

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, rand

import lumen
import lumen.functional as F


def _bf16(a):
    """``a`` rounded to bfloat16 and back, as float32."""
    return lumen.to_numpy(lumen.from_numpy(a).to(dtype="bfloat16").to(dtype="float32"))


def _compile(fn, device, preserve):
    """``fn`` compiled for ``device`` with ``preserve_precision`` set to ``preserve``."""
    config = getattr(lumen.config.compiler, device)
    config.preserve_precision = preserve
    return lumen.compile(fn, device=device)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_a_forward_round_trip_is_dropped(device, triton_on_cuda):
    """``x.bfloat16().float() * 3`` of a float32 ``x``: with
    ``preserve_precision`` the forward reads ``x`` itself, so the result is
    ``x * 3`` exactly; off (the default), ``x`` rounded to bfloat16 first, as
    traced."""
    x = rand(1000) * 7
    try:
        xt = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(x):
        """Three times ``x`` after its round trip through bfloat16."""
        return x.bfloat16().float() * 3.0

    try:
        assert not getattr(lumen.config.compiler, device).preserve_precision
        traced = lumen.to_numpy(_compile(f, device, False)(xt).to("cpu"))
        kept = lumen.to_numpy(_compile(f, device, True)(xt).to("cpu"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_array_equal(traced, _bf16(x) * np.float32(3))
    np.testing.assert_array_equal(kept, x * np.float32(3))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_gradients_keep_the_activation_saved_in_the_narrower_dtype(device, triton_on_cuda):
    """``sum(h * h)`` of ``h = x.bfloat16().float()``, with its gradient: with
    ``preserve_precision`` the loss is computed from ``x`` unrounded
    (``sum(x²)``), while the gradient reads ``h`` as saved for it, in
    bfloat16 (``2 · bf16(x)``), as mixed precision saves activations."""
    x = rand(512, seed=1)
    try:
        xt = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def loss(x):
        """The sum of squares of ``x`` after its round trip through bfloat16."""
        h = x.bfloat16().float()
        return F.sum(h * h)

    def step(x):
        """The loss and its gradient."""
        return loss(x), lumen.grad(loss)(x)

    try:
        value, gradient = (lumen.to_numpy(t.to("cpu")) for t in _compile(step, device, True)(xt))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_allclose(value, np.sum(x.astype(np.float64) ** 2), rtol=1e-5)
    np.testing.assert_array_equal(gradient, 2 * _bf16(x))
