"""Tests for ``F.cumsum`` and ``prims.cumsum`` on every device: values, reverse scans, dtypes, gradients and determinism."""

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU

import lumen
import lumen.functional as F
from lumen import prims

DEVICES = ["cpu", MPS, CUDA, TPU]


@pytest.fixture(autouse=True)
def triton():
    """CUDA's programs on the Triton backend."""
    lumen.config.compiler.cuda.use_triton_backend = True
    yield
    lumen.config.compiler.cuda.use_triton_backend = False


# (shape, axis) pairs: long rows split into chunks, tall columns, strided middle axes, a negative axis and an empty tensor.
CASES = [((4, 1000), 1), ((1, 70000), 1), ((3000, 3), 0), ((3, 50, 7), 1), ((8, 3), -1), ((5, 0, 3), 1)]


def _cumsum(x, axis, reverse):
    """``x``'s cumulative sum along ``axis`` in float64, from its end if
    ``reverse``."""
    x = x.astype(np.float64)
    return np.flip(np.cumsum(np.flip(x, axis), axis), axis) if reverse else np.cumsum(x, axis)


def _tensor(a, device, dtype):
    """Returns numpy array ``a`` as a ``dtype`` tensor on ``device``, skipping the test when the device is unavailable."""
    try:
        return lumen.from_numpy(a).to(device).to(dtype=dtype)
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "int32", "bfloat16"])
@pytest.mark.parametrize("shape, dim", CASES)
def test_cumsum(device, dtype, shape, dim):
    """``F.cumsum`` as numpy's: integers exactly, floats to float32 rounding
    (a bfloat16 tensor with ``dtype=float32`` widened as read, its result
    float32)."""
    a = (np.random.default_rng(0).standard_normal(shape) * 3).astype(np.float32)
    if dtype == "int32":
        a = a.astype(np.int32)
    x = _tensor(a, device, dtype)
    accum = "float32" if dtype == "bfloat16" else None
    y = lumen.compile(lambda x: F.cumsum(x, dim, dtype=accum))(x)
    assert y.dtype == (accum or dtype) and tuple(y.shape) == shape
    want = _cumsum(lumen.to_numpy(x.to(dtype="float32")), dim, False)
    got = lumen.to_numpy(y)
    if dtype == "int32":
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5 * max(1.0, np.abs(want).max(initial=0)))


@pytest.mark.parametrize("device", DEVICES)
def test_reverse_cumsum(device):
    """``prims.cumsum(x, axis, reverse=True, ...)`` sums from the axis's end."""
    a = np.random.default_rng(1).standard_normal((3, 2000)).astype(np.float32)
    x = _tensor(a, device, "float32")
    for axis in (0, 1):
        y = lumen.compile(lambda x: prims.cumsum(x, axis, True, "float32"))(x)
        np.testing.assert_allclose(lumen.to_numpy(y), _cumsum(a, axis, True), rtol=1e-5, atol=1e-4)


def test_cumsum_types():
    """It accumulates in the tensor's dtype, or float32 for half and
    bfloat16; ``dtype`` of another converts it first; bool is rejected."""
    m = lambda dtype: lumen.empty([4, 8], dtype=dtype, device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x: F.cumsum(x, 1, dtype="float32"))(m("bfloat16"))
    assert "cumsum[axis=1 reverse=False accum_dtype=f32]" in str(g) and "cast" not in str(g)
    g = lumen.make_graph(lambda x: F.cumsum(x, 0, dtype="float32"))(m("int32"))
    assert "cast" in str(g)
    with pytest.raises(ValueError, match="bool"):
        lumen.make_graph(lambda x: prims.cumsum(x, 0, False, "bool"))(m("bool"))
    with pytest.raises(ValueError, match="accum"):
        lumen.make_graph(lambda x: prims.cumsum(x, 0, False, "float32"))(m("int32"))


@pytest.mark.parametrize("device", DEVICES)
def test_cumsum_gradient(device):
    """cumsum is linear: the gradient of ``sum(cumsum(x) * w)`` is ``w``
    summed the other way, ``cumsum(w, reverse=True)``; and a bfloat16
    tensor's, accumulated in float32, is bfloat16."""
    rng = np.random.default_rng(2)
    a, w = (rng.standard_normal((3, 500)).astype(np.float32) for _ in range(2))
    x, wt = _tensor(a, device, "float32"), _tensor(w, device, "float32")
    grad = lumen.compile(lumen.grad(lambda x, w: F.sum(F.cumsum(x, 1) * w), 0))(x, wt)
    np.testing.assert_allclose(lumen.to_numpy(grad), _cumsum(w, 1, True), rtol=1e-5, atol=1e-4)
    xb = _tensor(a, device, "bfloat16")
    grad = lumen.compile(lumen.grad(lambda x, w: F.sum(F.cumsum(x, 1, dtype="float32") * w), 0))(xb, wt)
    assert grad.dtype == "bfloat16"
    want = _cumsum(w, 1, True)
    np.testing.assert_allclose(
        lumen.to_numpy(grad.to(dtype="float32")), want, rtol=2e-2, atol=2e-2 * np.abs(want).max()
    )


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_cumsum_is_deterministic(device):
    """Every scan adds in a fixed order (on MPS split ones too; on Triton a
    chunked scan with a carried sum; no atomics): the same bits each run."""
    x = _tensor(np.random.default_rng(3).standard_normal((2, 300000)).astype(np.float32), device, "float32")
    f = lumen.compile(lambda x: F.cumsum(x, 1))
    first = lumen.to_numpy(f(x))
    for _ in range(5):
        np.testing.assert_array_equal(lumen.to_numpy(f(x)), first)
