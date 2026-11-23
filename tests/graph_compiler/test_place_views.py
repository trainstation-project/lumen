"""Tests of slices read in place: a matmul reading a slice of its operand's value instead of a copy of it."""

import numpy as np
import pytest
from helpers import TPU, rand

import lumen
import lumen.functional as F


def _mm(a, b):
    """``a @ b`` accumulated and returned in float32."""
    return F.matmul(a, b, "float32", "float32")


@pytest.mark.parametrize("device", [TPU], indirect=True)
@pytest.mark.parametrize(
    "f, shapes, steps",
    [
        (lambda x, w: _mm(x[64:192], w.t()), [(256, 512), (256, 512)], ["dot_general"]),
        (lambda x, w: _mm(x, w[:, 128:384]), [(256, 512), (512, 512)], ["dot_general"]),
        (lambda x, w: _mm(x, w[128:384].t()), [(256, 512), (512, 512)], ["dot_general"]),
        (lambda x, w: _mm(x[:, 128:256], w[256:384]), [(256, 512), (512, 256)], ["dot_general"]),
        (lambda x, w: _mm(x[:100], w), [(256, 256), (256, 128)], ["dot_general"]),
        (lambda x, w: _mm(x[..., 128:384], w), [(2, 64, 512), (256, 128)], ["dot_general", "reshape"]),
        (
            lambda x, w, b: F.relu(_mm(x, w[:, 256:512]) + b),
            [(128, 256), (256, 768), (128, 256)],
            ["dot_general → add → max"],
        ),
        (lambda x, w: _mm(x[:, :256], w) + F.exp(x[:, :256]), [(128, 512), (256, 256)], None),
    ],
    ids=["lhs-rows", "rhs-columns", "rhs-rows-transposed", "both", "first-rows", "batched", "epilogue", "shared"],
)
def test_tpu_matmuls_read_slices_in_place(device, f, shapes, steps):
    """On TPU, a slice read only by matmuls, of whole tiles of its value
    (rows of whole 8- or 16-row tiles, a matrix's first rows, columns of
    whole 128-lane tiles), is no copy: the matmul's blocks step through
    the value from the slice's start. One also read elsewhere stays a
    slice of its own. The CPU's values."""
    arrays = [rand(*s, seed=k) / 8 for k, s in enumerate(shapes)]
    try:
        args = [lumen.from_numpy(v).to(device) for v in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]
    if steps is None:
        assert "slice" in labels, labels
    else:
        assert labels == steps
    got = lumen.to_numpy(lumen.compile(f)(*args).to("cpu"))
    want = lumen.to_numpy(lumen.compile(f)(*(lumen.from_numpy(v) for v in arrays)))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", [TPU], indirect=True)
def test_tpu_bfloat16_matmul_reads_a_slice_in_place(device):
    """A bfloat16 weight's column slice is read in place too (16-row
    tiles), the cast to bfloat16 the matmul's epilogue."""
    x, w = rand(64, 256) / 8, rand(256, 512, seed=1) / 8
    try:
        args = [lumen.from_numpy(v).to(dtype="bfloat16").to(device) for v in (x, w)]
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(x, w):
        """x times a column slice of w, in bfloat16."""
        return F.matmul(x, w[:, 128:256], "float32", "bfloat16")

    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]
    assert labels == ["dot_general → cast(float32 -> bfloat16)"]
    got = lumen.to_numpy(lumen.compile(f)(*args).to("cpu").to(dtype="float32"))
    want = lumen.to_numpy(lumen.compile(f)(*(a.to("cpu") for a in args)).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)
