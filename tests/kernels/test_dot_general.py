"""Tests of matmul kernels: dots against numpy in every layout, vector dots, folded batches, tiles and tuning."""

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, floats, placed

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile


@pytest.mark.parametrize("device", ["cpu", MPS, TPU, CUDA])
@pytest.mark.parametrize(
    "lhs, rhs, dtype, transposed, output",
    [
        ([256, 512], [512, 384], "float32", False, None),
        ([64, 1024], [1024, 128], "float32", False, None),
        ([1, 256], [256, 256], "float32", False, None),
        ([2, 3, 256], [256, 128], "float32", False, None),
        ([2, 64, 256], [2, 256, 384], "float32", False, None),
        ([4, 8], [8, 16], "float32", False, None),
        ([3, 100], [100, 30], "bfloat16", False, None),
        ([64, 256], [384, 256], "float32", True, None),
        ([8, 1024], [512, 1024], "bfloat16", True, None),
        ([8, 1024], [512, 1024], "bfloat16", True, "float32"),
    ],
)
def test_dots_agree_with_numpy(device, lhs, rhs, dtype, transposed, output, triton_on_cuda):
    """A dot, in float32 on the operands as placed, within its operands'
    dtype's precision of its largest output (float32's when it writes
    float32)."""
    x, w = placed(lhs, dtype, device), placed(rhs, dtype, device, seed=1)
    out = output or dtype
    y = lumen.compile(lambda x, w: F.matmul(x, w.t() if transposed else w, "float32", out))(x, w)
    assert y.device == f"{device}:0" and y.dtype == out
    a, b = floats(x), floats(w)
    expected = a @ (b.T if transposed else b)
    assert list(y.shape) == list(expected.shape)
    tol = 1e-5 if out == "float32" else 1e-2
    assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_dot_of_two_vectors_is_a_product_and_a_sum(device, dtype, triton_on_cuda):
    """``rowsum(dO * O)`` as attention's backward traces it, a dot with no
    free dimensions: dot strength reduction's product and row sum, in
    float32."""
    x, y = placed([2, 8, 1024, 128], dtype, device), placed([2, 8, 1024, 128], dtype, device, seed=1)
    out = lumen.compile(lambda x, y: F.einsum("bhsd,bhsd->bhs", x, y, accum_dtype="float32", output_dtype=x.dtype))(
        x, y
    )
    assert out.device == f"{device}:0" and list(out.shape) == [2, 8, 1024]
    expected = (floats(x) * floats(y)).sum(-1)
    assert np.abs(floats(out) - expected).max() <= {"float32": 1e-5, "bfloat16": 1e-2}[dtype] * np.abs(expected).max()


def test_matmul_folds_a_batch_times_a_matrix():
    """A batch times a matrix (``x @ w``, a weight) is one matmul of the
    batch's rows, as torch folds it: the matrix read as it is, never
    broadcast over the batch (a copy); its gradient one matmul too, not a
    sum of the batch's."""
    m = lambda *s: lumen.empty(list(s), device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, w: F.matmul(x, w, "float32", x.dtype))(m(2, 3, 4), m(4, 5))
    assert "broadcast_in_dim" not in str(g) and "f32[6,5] = dot_general" in str(g), g
    grad = lumen.make_graph(lumen.grad(lambda x, w: F.sum(F.matmul(x, w, "float32", x.dtype)), 1))(m(2, 3, 4), m(4, 5))
    assert str(grad).count("dot_general") == 1 and "reduce_sum" not in str(grad), grad


@pytest.mark.mps
@pytest.mark.parametrize("m, tiles", [(64, "mid"), (192, "mid"), (16, "small"), (256, "large")])
def test_matmul_tiles_fit_its_rows(m, tiles):
    """Untuned (``autotune`` off), a float matmul runs on 128x64 tiles; on
    64x64 ones where M leaves 33 to 64 rows of a last one (M of 64, 192:
    half a large tile idle, the 32x32 tiles reading the operands twice as
    often); on 32x32 where it leaves 32 or fewer (or the large tiles are too
    few to fill the GPU). With an epilogue too (its kernel's ``_mid``,
    ``_small`` variant). Tuned, it runs on whichever tile was fastest. The
    results NumPy's either way."""
    rng = np.random.default_rng(0)
    k, n = 256, 2048
    try:
        a, b = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((m, k), (k, n))
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    product = lumen.to_numpy(a.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(b.to(dtype="float32"))

    def plain(a, b):
        """float32 matmul."""
        return F.matmul(a, b, "float32", "float32")

    def fused(a, b):
        """relu of the float32 matmul."""
        return F.relu(plain(a, b))

    for tuned in (False, True):
        lumen.config.compiler.mps.autotune = tuned
        try:
            for f, want in ((plain, product), (fused, np.maximum(product, 0))):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
                    got = lumen.to_numpy(lumen.compile(f)(a, b))
                (kernel,) = {e["kernel"] for e in prof.events() if e["kind"] == "gpu"}
                ran = "small" if "small" in kernel else "mid" if "mid" in kernel else "large"
                assert tuned or ran == tiles, kernel
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
        finally:
            lumen.config.compiler.reset()


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lhs_k, rhs_k", [(1, 0), (1, 1), (0, 0), (0, 1)])
@pytest.mark.parametrize("m, k, n", [(512, 256, 512), (300, 200, 260), (5, 4096, 1000), (1, 4096, 4096)])
def test_tuned_matmul_matches_numpy_in_every_layout(device, dtype, lhs_k, rhs_k, m, k, n, triton_on_cuda):
    """Tuned, a float matmul of 64K or more outputs, or of a few rows,
    runs on whichever candidate was fastest: lumen's simdgroup kernels,
    where Metal Performance Primitives compile ``matmul_mpp``, and for a
    few rows ``matmul_skinny``. Either operand may be transposed
    (contracting its first dimension rather than its last, or its last
    rather than its first), tiles may be left partial at the edges (300 x
    260 of 64-wide tiles, 5 rows of 8-row ones), and the result may round
    to bfloat16. Each matches NumPy, on CUDA (Triton's GEMM) too."""
    rng = np.random.default_rng(0)
    shapes = ((m, k) if lhs_k == 1 else (k, m), (k, n) if rhs_k == 0 else (n, k))
    try:
        a, b = (lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to(device).to(dtype=dtype) for s in shapes)
    except RuntimeError as e:
        pytest.skip(str(e))
    x, y = (lumen.to_numpy(t.to(dtype="float32")).astype(np.float64) for t in (a, b))
    want = (x if lhs_k == 1 else x.T) @ (y if rhs_k == 0 else y.T)
    dims = (((lhs_k,), (rhs_k,)), ((), ()))
    got = lumen.compile(lambda a, b: prims.dot_general(a, b, dims, "float32", dtype))(a, b)
    rtol = 1e-4 if dtype == "float32" else 2**-6
    np.testing.assert_allclose(lumen.to_numpy(got.to(dtype="float32")), want, rtol=rtol, atol=rtol * 16)
