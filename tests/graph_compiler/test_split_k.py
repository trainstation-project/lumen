"""Tests of split-K: long-K matmuls with few output tiles run as partial products over K chunks, then their sum."""

import numpy as np
import pytest
from helpers import CUDA, MPS

import lumen
import lumen.functional as F


@pytest.fixture
def compiler(device, triton_on_cuda):
    """``device``'s compiler settings (CUDA's running Triton), reset to their defaults after the test."""
    yield getattr(lumen.config.compiler, device)
    lumen.config.compiler.reset()


def _labels(f, device, *args):
    """Step labels of `f` traced on `args` and planned for `device`."""
    return [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]


def _placed(device, dtype, *arrays):
    """`arrays` as `dtype` tensors on `device`, skipping the test when it is unavailable."""
    try:
        return [lumen.from_numpy(a).to(device).to(dtype=dtype) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_split_k_agrees(device, compiler, dtype):
    """With ``split_k`` (the default) a matmul launching few tiles and of a
    long contraction (a decode step's) is one dot of the chunks' partials,
    then their sum (each labelled split-K), rounded to bfloat16 once, after
    it. Each agrees with the dot as traced (off) to rounding. One of many
    tiles is not split. On MPS the partials may be added atomically (not
    ``deterministic``) where they cost more than the operands: one kernel
    adding its chunks' products to its float32 output; Triton always sums
    them in a reduction."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 4096)).astype(np.float32), rng.standard_normal((4096, 256)).astype(np.float32)
    x, w = _placed(device, dtype, a, b)
    f = lambda x, w: F.matmul(x, w, "float32", x.dtype)  # noqa: E731
    want = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    cast = "cast(float32 -> bfloat16)"
    rounded = {"float32": "", "bfloat16": " → " + cast}[dtype]
    split_plan = ["dot_general (split-K)", "reduce_sum (split-K)" + rounded]
    assert _labels(f, device, x, w) == split_plan
    results = [lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))]
    if device == "mps":
        compiler.deterministic = True
        assert _labels(f, device, x, w) == split_plan
        results.append(lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32")))
    compiler.split_k = False
    assert _labels(f, device, x, w) == ["dot_general" + rounded]
    results.append(lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32")))
    tol = {"float32": 1e-4, "bfloat16": 2e-2}[dtype] * np.abs(want).max()
    for got in results:
        np.testing.assert_allclose(got, want, atol=tol)
    compiler.split_k = True
    if device == "mps":
        compiler.deterministic = False
    big = lumen.empty([1024, 1024], device="meta")
    assert _labels(f, device, big, big) == ["dot_general"]
    if device != "mps":
        return
    a, b = (rng.standard_normal(s).astype(np.float32) for s in ((352, 2048), (2048, 352)))
    x, w = _placed(device, "bfloat16", a, b)
    want = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    assert _labels(f, device, x, w) == ["dot_general (split-K)", cast]
    atomic = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
    np.testing.assert_allclose(atomic, want, atol=2e-2 * np.abs(want).max())
    compiler.deterministic = True
    assert _labels(f, device, x, w) == ["dot_general (split-K)", "reduce_sum (split-K) → " + cast]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_split_k_then_silu(device, compiler, dtype):
    """SiLU after a matmul of few output tiles and a long contraction: not
    split, it is the matmul's epilogue (one kernel); split, it needs every
    chunk's products: the epilogue of the ``reduce_sum`` of the dot's
    partials (reading the sum twice), on MPS atomic adds allowed or not
    (its partials cost less than its operands). Each agrees with SiLU of
    the exact product to rounding."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 4096)).astype(np.float32), rng.standard_normal((4096, 256)).astype(np.float32)
    x, w = _placed(device, dtype, a, b)

    def f(x, w):
        """SiLU of a float32-accumulated matmul, cast back to the input dtype."""
        y = F.matmul(x, w, accum_dtype=lumen.float32, output_dtype=lumen.float32)
        return (y * F.sigmoid(y)).to(dtype=x.dtype)

    y = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    want = y / (1 + np.exp(-y))
    silu = "logistic → mul" + {"float32": "", "bfloat16": " → cast(float32 -> bfloat16)"}[dtype]
    plans = {
        "unsplit": ["dot_general → " + silu],
        "split": ["dot_general (split-K)", "reduce_sum (split-K) → " + silu],
    }
    if device == "mps":
        plans["deterministic"] = plans["split"]
    tol = {"float32": 1e-4, "bfloat16": 2e-2}[dtype] * np.abs(want).max()
    for mode, plan in plans.items():
        compiler.split_k = mode != "unsplit"
        if device == "mps":
            compiler.deterministic = mode == "deterministic"
        assert _labels(f, device, x, w) == plan, mode
        got = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
        np.testing.assert_allclose(got, want, atol=tol, err_msg=mode)
