"""Tests of in-place concatenation: operands written straight into their slot of the result."""

import numpy as np
import pytest
from helpers import CUDA, MPS, rand

import lumen
import lumen.functional as F


def _matmuls(a, b, w1, w2):
    """Two independent matmuls concatenated along their columns (strided slots)."""
    return F.cat([F.matmul(a, w1, "float32", "float32"), F.matmul(b, w2, "float32", "float32")], -1)


def _loops(a, b):
    """Two elementwise loops of different sizes concatenated along rows."""
    return F.cat([F.exp(a), b.t() * 2.0], 0)


@pytest.mark.parametrize("device", [MPS, CUDA])
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
            {"mps": ["exp → mul → transpose → concatenate"], "cuda": ["exp", "mul → transpose"]},
        ),
    ],
    ids=["matmuls", "loops"],
)
def test_concatenated_operands_are_written_in_place(device, f, shapes, steps, triton_on_cuda):
    """On Triton, a concatenate whose operands are each read only there and
    produced by a kernel of their own (a matmul, a loop) copies nothing: each
    kernel writes its operand into its slot of the result through the slot's
    strides (the loops as two: the transpose runs through tiles, which
    horizontal fusion keeps apart, as on MPS). MPS places only what
    attention's backward writes, and runs the others as a copy (two matmuls')
    or inside the loop computing the operands. The CPU's values either way."""
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
