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


@pytest.mark.mps
def test_mps_dot_of_two_dtypes_is_refused():
    """MPS has no dot of operands of two dtypes: compiling one raises
    naming both and the way out; the CPU reference computes it."""
    a, b = np.ones((4, 8), np.float32), np.ones((8, 3), np.float32)
    f = lambda a, b: prims.dot_general(a, b, (([1], [0]), ([], [])), "float32", "float32")  # noqa: E731
    cpu = lumen.compile(f, device="cpu")(lumen.from_numpy(a), lumen.from_numpy(b).to(dtype="float16"))
    np.testing.assert_array_equal(lumen.to_numpy(cpu), np.full((4, 3), 8, np.float32))
    try:
        x, y = lumen.from_numpy(a).to("mps"), lumen.from_numpy(b).to("mps").to(dtype="float16")
    except RuntimeError as e:
        pytest.skip(str(e))
    with pytest.raises(
        RuntimeError, match="dot_general: has no kernel of two dtypes on MPS, got f32\\[4,8\\] and f16"
    ):
        lumen.compile(f, device="mps")(x, y)


@pytest.mark.mps
@pytest.mark.parametrize("n, vocab, tiles", [(192, 2048, "mid"), (256, 2048, "large"), (5, 7, "small"), (0, 7, None)])
def test_mps_linear_cross_entropy_tiles_and_launches(n, vocab, tiles):
    """MPS's linear cross entropy (``cross_entropy(h @ w.T)``, nothing else
    reading the logits) runs its matmul on the tiles that fit its rows (64
    x 64 where 33 to 64 rows are left of a 128-row tile) and is profiled as
    its two launches, partials (each row's max and sum of exps a tile of
    classes, ``[2, n, tiles]``) then combine (into ``[2, n]``: losses and
    log-sum-exps); of no rows it launches nothing. Its losses NumPy's."""
    from lumen import functional as F

    rng = np.random.default_rng(0)
    h = (rng.standard_normal((n, 32)) / 4).astype(np.float32)
    w = rng.standard_normal((vocab, 32)).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    try:
        H, W, T = (lumen.from_numpy(a).to("mps") for a in (h, w, t))
    except RuntimeError as e:
        pytest.skip(str(e))

    def loss(h, w, t):
        """Per-row cross entropy of the logits `h @ w.T`."""
        return F.cross_entropy(F.matmul(h, w.t(), "float32", "float32"), t)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        got = lumen.to_numpy(lumen.compile(loss)(H, W, T))
    kernels = [e["kernel"] for e in prof.events() if e["kind"] == "gpu"]
    if tiles is None:
        assert got.shape == (0,) and not any(k.startswith("lce_") for k in kernels), kernels
        return
    launches = [e for e in prof.events() if e["name"].startswith("linear_cross_entropy (")]
    assert [e["name"] for e in launches] == ["linear_cross_entropy (partials)", "linear_cross_entropy (combine)"]
    (_, (_, partial)), (_, (_, losses)) = ((e["inputs"], e["outputs"][0]) for e in launches)
    assert partial[:2] == [2, n] and launches[1]["inputs"] == [("float32", partial)] and losses == [2, n], launches
    partials = next(k for k in kernels if k.startswith("lce_partials"))
    assert ("small_" in partials, "mid_" in partials) == (tiles == "small", tiles == "mid"), partials
    x = h.astype(np.float64) @ w.T
    m = x.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
    np.testing.assert_allclose(got, lse - x[np.arange(n), t], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("norm", ["rms_norm", "softmax"])
def test_a_vector_times_a_matrix_fused_with_a_wide_norm(device, norm, triton_on_cuda):
    """A decode step's projection of one row (``x @ w.T``, its K
    contiguous: a matrix-vector product) normalized over 4096 columns,
    more than one tile: on MPS one fusion whose matmul is the
    matrix-vector kernel over 64-column tiles, then the row's norm.
    Values the CPU's."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 256)).astype(np.float32)
    w = (rng.standard_normal((4096, 256)) / 16).astype(np.float32)

    def f(x, w):
        """The norm of the row's projection."""
        y = F.matmul(x, w.t(), "float32", "float32")
        return F.rms_norm(y, [4096], None, 1e-6) if norm == "rms_norm" else F.softmax(y, -1)

    want = lumen.to_numpy(lumen.compile(f, device="cpu")(lumen.from_numpy(x), lumen.from_numpy(w)))
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps" and norm == "rms_norm":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), "mps").steps()
        assert step["label"].startswith("dot_general"), step["label"]
        assert "matmul_gemv" in step["fusion"]["source"], step["label"]
    got = lumen.to_numpy(lumen.compile(f)(X, W).to("cpu"))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
