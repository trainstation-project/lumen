"""Tests of in-place concatenation: operands written straight into their slot of the result."""

import numpy as np
import pytest
from helpers import MPS, TPU, TRITON_CUDA, rand

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


def _exp_first(x, y, axis):
    """The concatenation along ``axis`` of ``exp(x)`` and the input ``y``, which no kernel makes."""
    return F.cat([F.exp(x), y], axis)


def _dots(x, a, b):
    """Two matmuls of ``x`` concatenated along their columns, the second's slot a block past the first."""
    return F.cat([F.matmul(x, a, "float32", "float32"), F.matmul(x, b, "float32", "float32")], -1)


def _scaled(x, y):
    """Two elementwise fusions concatenated along their columns."""
    return F.cat([F.exp(x), y * 2.0], -1)


@pytest.mark.parametrize("device", [TPU], indirect=True)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "f, shapes, steps",
    [
        (lambda x, y: _exp_first(x, y, -1), [(64, 256), (64, 128)], ["exp", "concatenate"]),
        (lambda x, y: _exp_first(x, y, -1), [(64, 192), (64, 64)], ["exp", "concatenate"]),
        (lambda x, y: _exp_first(x, y, 0), [(32, 128), (16, 128)], ["exp", "concatenate"]),
        (lambda x, y: _exp_first(x, y, 0), [(12, 256), (20, 256)], ["exp", "concatenate"]),
        (lambda x, y: _exp_first(x, y, 1), [(2, 3, 16, 128), (2, 1, 16, 128)], ["exp", "concatenate"]),
        (_scaled, [(64, 256), (64, 384)], ["exp", "mul"]),
        (_dots, [(64, 256), (256, 256), (256, 384)], ["dot_general", "dot_general"]),
    ],
    ids=["lanes", "lanes-unaligned", "rows", "rows-unaligned", "leading", "elementwise-slots", "dot-slots"],
)
def test_tpu_concatenates(device, dtype, f, shapes, steps):
    """On TPU, a concatenation its operands' kernels can't write in place
    (an input, an axis other than the last, a width of no whole lanes) is
    copied by a kernel of its own: DMAs into slots starting on tiles, else
    blocks of the result built from its operands'. Operands kernels make
    along the last axis are written straight into their slots, whose
    blocks divide the result's width. The CPU's values."""
    if dtype == "bfloat16" and f is _dots:
        pytest.skip("the float32 matmul takes float32 operands")
    arrays = [rand(*s, seed=k) / 8 for k, s in enumerate(shapes)]
    args = [lumen.from_numpy(v).to(dtype=dtype) for v in arrays]
    try:
        on = [a.to(device) for a in args]
    except RuntimeError as e:
        pytest.skip(str(e))
    plan = lumen.graph_compiler.Plan(lumen.make_graph(f)(*on), device)
    assert [s["label"] for s in plan.steps()] == steps
    got = lumen.to_numpy(lumen.compile(f)(*on).to("cpu").to(dtype="float32"))
    want = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)
