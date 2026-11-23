"""Tests of in-place concatenation: operands written straight into their slot of the result."""

import numpy as np
import pytest
from helpers import MPS, TRITON_CUDA, rand

import lumen
import lumen.functional as F


def _matmuls(a, b, w1, w2):
    """Two independent matmuls concatenated along their columns (strided slots)."""
    return F.cat([F.matmul(a, w1, "float32", "float32"), F.matmul(b, w2, "float32", "float32")], -1)


def _loops(a, b):
    """Two elementwise fusions of different sizes concatenated along rows."""
    return F.cat([F.exp(a), b.t() * 2.0], 0)


@pytest.mark.parametrize("device", [MPS, TRITON_CUDA], indirect=True)
@pytest.mark.parametrize(
    "f, shapes, steps",
    [
        (
            _matmuls,
            [(64, 96), (64, 48), (96, 80), (48, 40)],
            {"mps": ["dot_general", "dot_general", "concatenate"], "cuda": ["dot_general", "dot_general"]},
        ),
        (
            _loops,
            [(64, 32), (32, 16)],
            {"mps": ["exp → mul → transpose → concatenate"], "cuda": ["exp → mul → transpose → concatenate"]},
        ),
    ],
    ids=["matmuls", "elementwise"],
)
def test_concatenated_operands_are_written_in_place(device, f, shapes, steps):
    """A concatenate whose operands are each read only there and made by a
    kernel of their own (a matmul) copies nothing on Triton: each matmul
    writes its operand into its slot of the result through the slot's
    strides; MPS's matmuls write no slots, so it copies them. Operands made
    by cheap elementwise ops aren't placed on either: the concatenate runs
    inside the elementwise fusion computing them (MPS and Triton share the
    rule, `place_concats::worth`). The CPU's values either way."""
    arrays = [rand(*s, seed=k) for k, s in enumerate(shapes)]
    try:
        args = [lumen.from_numpy(v).to(device) for v in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    plan = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device)
    assert [s["label"] for s in plan.steps()] == steps[device]
    got = lumen.to_numpy(lumen.compile(f)(*args))
    want = lumen.to_numpy(lumen.compile(f)(*(lumen.from_numpy(v) for v in arrays)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5 * np.abs(want).max())


def _rotary(x, cos, sin):
    """A rotary-like rotation of the last axis by a quarter (not a half: `swap_pairs` would take that) and a
    multiply-add: its concatenate of slices and a negation is read only by elementwise ops."""
    quarter = x.shape[-1] // 4
    rotated = F.cat([-x[..., 3 * quarter :], x[..., : 3 * quarter]], -1)
    return x * cos + rotated * sin


@pytest.mark.parametrize("device", [MPS, TRITON_CUDA], indirect=True)
def test_concatenates_read_elementwise_fuse_into_their_readers(device):
    """A concatenate of cheap operands (slices, a negation) read only by
    elementwise ops is not placed: fusion builds it in registers inside its
    readers' kernel, so the rotation and multiply-add are one kernel
    with no concatenate step and no buffer for it. The CPU's values."""
    arrays = [rand(4, 8, 64, seed=k) for k in range(3)]
    try:
        args = [lumen.from_numpy(v).to(device) for v in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    plan = lumen.graph_compiler.Plan(lumen.make_graph(_rotary)(*args), device)
    (label,) = [s["label"] for s in plan.steps()]
    assert "concatenate" in label and "neg" in label and label.endswith("add"), label
    got = lumen.to_numpy(lumen.compile(_rotary)(*args))
    want = lumen.to_numpy(lumen.compile(_rotary)(*(lumen.from_numpy(v) for v in arrays)))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
