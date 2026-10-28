"""Tests of cross entropy: one row kernel, its gradient, and linear and chunked cross entropy of a matmul's logits."""

import re

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU

import lumen
import lumen.functional as F


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n, vocab", [(64, 16), (16, 16), (300, 1000), (128, 32000)])
def test_cross_entropy_is_one_row_kernel(device, n, vocab, triton_on_cuda):
    """``F.cross_entropy`` (``reduction="none"``): each row's ``logsumexp -
    logit[label]``. On MPS one row kernel reads the logits once (an online
    max and sum, and the label's logit picked by a one-hot sum, all one
    loop over the row: none reads another's value), writing a loss a row.
    Trained (``g`` not read from the losses), its gradient ``(exp(x - lse)
    - one_hot) · g`` is the same kernel's second pass over the row (XMA's
    forward-backward kernel), the losses its other output: one kernel,
    nothing of the logits' shape but the gradient written (its one-hot's
    classes along the row, as many as the rows or not). Losses and
    gradients NumPy's."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    c = rng.standard_normal(n).astype(np.float32)

    def step(x, t, c):
        """Per-row losses and the logits' gradient of their weighted sum."""
        loss = F.cross_entropy(x, t)
        F.sum(loss * c).backward()
        return loss, x.grad

    try:
        X, T, C = (lumen.from_numpy(a).to(device) for a in (x, t, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        (forward,) = lumen.graph_compiler.Plan(lumen.make_graph(F.cross_entropy)(X, T), "mps").steps()
        assert forward["output"][2] == [n] and "reduce_max" in forward["label"], forward["label"]
        source = forward["fusion"]["source"]
        assert source.count("for (uint c") + source.count("for (uint e = 0") == 1, source
        steps = lumen.graph_compiler.Plan(lumen.make_graph(step)(X, T, C), "mps").steps()
        assert [s["output"][2] for s in steps] == [[n, vocab]], [s["label"] for s in steps]
        assert [o[2] for o in steps[0]["extra_outputs"]] == [[n]], steps[0]["label"]
    m = x.max(-1, keepdims=True).astype(np.float64)
    lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
    hit = np.zeros_like(x, dtype=np.float64)
    hit[np.arange(n), t] = 1
    loss, grad = (lumen.to_numpy(v) for v in lumen.compile(step)(X, T, C))
    np.testing.assert_allclose(loss, lse - x[np.arange(n), t], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grad, (np.exp(x - lse[:, None]) - hit) * c[:, None], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("rows_read", ["its own", "every row's"])
@pytest.mark.parametrize("n, vocab", [(64, 16), (128, 32000)])
def test_cross_entropy_gradient_reading_the_losses(device, n, vocab, rows_read, triton_on_cuda):
    """A cross entropy whose gradient's ``g`` reads the losses. Each row's
    its own (``Σ loss²``, ``g = 2·loss``): computed once a row in its row
    kernel after the reductions, the gradient its second pass (one
    kernel). Every row's (``mean(loss)²``, ``g = 2·mean(loss)/n``): not
    known until every row's loss is, so the gradient is a pass after the
    row kernel (and the mean) reading ``lse``, which the row kernel writes
    beside the losses; never ``softmax - one_hot`` written to be scaled
    later. The gradients NumPy's."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((n, vocab)) * 3).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    of = {"its own": lambda loss: F.sum(loss * loss), "every row's": lambda loss: F.mean(loss) * F.mean(loss)}

    def step(x, t):
        """The losses and the logits' gradient of the case's loss reduction."""
        loss = F.cross_entropy(x, t)
        of[rows_read](loss).backward()
        return loss, x.grad

    try:
        X, T = (lumen.from_numpy(a).to(device) for a in (x, t))
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(step)(X, T), "mps").steps()
        labels = [s["label"] for s in steps]
        if rows_read == "its own":
            assert [s["output"][2] for s in steps] == [[n, vocab]], labels
        else:
            assert [s["output"][2] for s in steps] == [[n], [n], [n, vocab]], labels
            assert [o[2] for o in steps[0]["extra_outputs"]] == [[n]] and "reduce" not in steps[2]["label"], labels
    xr = x.astype(np.float64)
    m = xr.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(xr - m).sum(-1, keepdims=True)))[:, 0]
    hit = np.zeros_like(xr)
    hit[np.arange(n), t] = 1
    loss = lse - xr[np.arange(n), t]
    g = 2 * loss[:, None] if rows_read == "its own" else 2 * loss.mean() / n
    _, grad = (lumen.to_numpy(v) for v in lumen.compile(step)(X, T))
    np.testing.assert_allclose(grad, g * (np.exp(xr - lse[:, None]) - hit), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
@pytest.mark.parametrize(
    "n, d, vocab, dtype", [(5, 16, 7, "float32"), (300, 64, 1000, "float32"), (128, 256, 32000, "bfloat16")]
)
def test_linear_cross_entropy(device, n, d, vocab, dtype, triton_on_cuda):
    """``F.cross_entropy(F.matmul(h, w.t()), t)``, the MPS compiler's linear
    cross entropy where nothing else reads the logits (not trained): one
    step of two kernels: the matmul writing each row's
    max and sum of exps a tile of 64 classes (and the label's logit), then
    a kernel combining a row's tiles into its ``logsumexp``. Its losses and
    gradients NumPy's (of the operands as rounded to ``dtype``)."""
    if device == "cpu" and vocab > 1000:
        pytest.skip("slow on the CPU")
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    w = rng.standard_normal((vocab, d)).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    c = rng.standard_normal(n).astype(np.float32)

    def loss(h, w, t):
        """Per-row cross entropy of the logits `h @ w.t()`."""
        return F.cross_entropy(F.matmul(h, w.t(), "float32", "float32"), t)

    def step(h, w, t, c):
        """The losses and the gradients of `h` and `w` for their weighted sum."""
        losses = loss(h, w, t)
        F.sum(losses * c).backward()
        return losses, h.grad, w.grad

    try:
        H, W, T, C = (lumen.from_numpy(a).to(device) for a in (h, w, t, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    H, W = H.to(dtype=dtype), W.to(dtype=dtype)
    if device in ("mps", "cuda"):
        steps = lumen.graph_compiler.Plan(lumen.make_graph(loss)(H, W, T), device).steps()
        assert steps[0]["label"] == "linear_cross_entropy" and steps[0]["output"][2] == [2, n], steps
    hr, wr = (lumen.to_numpy(v.to(dtype="float32")).astype(np.float64) for v in (H, W))
    x = hr @ wr.T
    m = x.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
    hit = np.zeros_like(x)
    hit[np.arange(n), t] = 1
    dx = (np.exp(x - lse[:, None]) - hit) * c[:, None]
    losses, dh, dw = (lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(step)(H, W, T, C))
    np.testing.assert_allclose(losses, lse - x[np.arange(n), t], rtol=1e-5, atol=1e-5)
    tol = dict(rtol=1e-4, atol=1e-5) if dtype == "float32" else dict(rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(dh, dx @ wr, **tol)
    np.testing.assert_allclose(dw, dx.T @ hr, **tol)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n, d, vocab", [(64, 32, 16), (256, 128, 32000)])
def test_cross_entropy_of_a_matmul_is_linear_cross_entropy(device, n, d, vocab, triton_on_cuda):
    """``F.cross_entropy`` of a matmul's float32 logits (an LM head's, its
    weight transposed and cast), nothing else reading them: the MPS
    compiler matches it as ``linear_cross_entropy`` (its pass, as flash
    attention's), no step writing the logits. With the logits read too
    (returned), not matched. The losses NumPy's either way."""
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    w = rng.standard_normal((vocab, d)).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)

    def loss(h, w, t):
        """Per-row cross entropy of float32 logits from a bfloat16 head."""
        return F.cross_entropy(F.matmul(h, w.t().bfloat16(), "float32", "float32"), t)

    def both(h, w, t):
        """The losses and the logits too."""
        logits = F.matmul(h, w.t().bfloat16(), "float32", "float32")
        return F.cross_entropy(logits, t), logits

    try:
        H, W, T = (lumen.from_numpy(a).to(device) for a in (h, w, t))
    except RuntimeError as e:
        pytest.skip(str(e))
    H = H.to(dtype="bfloat16")
    steps = lumen.graph_compiler.Plan(lumen.make_graph(loss)(H, W, T), device).steps()
    assert "linear_cross_entropy" in [s["label"] for s in steps], [s["label"] for s in steps]
    assert all(s["output"][2] != [n, vocab] for s in steps), [s["label"] for s in steps]
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(both)(H, W, T), "mps").steps()
        assert "linear_cross_entropy" not in [s["label"] for s in steps], [s["label"] for s in steps]
    hr = lumen.to_numpy(H.to(dtype="float32")).astype(np.float64)
    wr = lumen.to_numpy(W.to(dtype="bfloat16").to(dtype="float32")).astype(np.float64)
    x = hr @ wr.T
    m = x.max(-1, keepdims=True)
    want = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0] - x[np.arange(n), t]
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(loss)(H, W, T)), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(both)(H, W, T)[0]), want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("reduce", ["mean", "weighted", "squared"])
@pytest.mark.parametrize(
    "n, d, vocab, chunk, head",
    [
        (64, 32, 16, 16, "bfloat16"),
        (1000, 256, 5000, 128, "bfloat16"),
        (1024, 64, 1984, 32, "float32"),
        (64, 32, 16, 64, "bfloat16"),
    ],
)
def test_chunked_linear_cross_entropy(device, n, d, vocab, chunk, reduce, head, triton_on_cuda):
    """A cross entropy of a matmul's logits trained with
    ``fused_linear_cross_entropy_chunk_size`` (MPS's or CUDA's)
    (``lumen/autograd/cross_entropy.py``): its backward a chunk of rows at
    a time (4 of 16 rows of 64; 8 of 128 rows of 1000, the last one
    short; 32 of 32 rows of 1024, more than a kernel binds, its losses
    joined a few chunks' at a time; or one chunk, all 64 rows: its logits
    the forward's, the program unchunked). Its weight bfloat16, or float32
    cast to bfloat16 in the step (an LM head's master weight, its gradient
    float32). On MPS and Triton each chunk's logits computed once (XMA's fused linear
    cross entropy): a matmul writing them, then one row kernel writing
    the chunk's losses and ``(softmax - one_hot) · g`` (any ``g``: a
    mean's, a weighted sum's) from them, then its rows' gradients; the
    forward's losses those chunks', the whole logits never computed. Not
    where ``g`` reads every row's loss (``mean(loss)²``): the chunk's
    logits recomputed after a linear cross entropy, its forward. The
    logits of a chunk at a time in memory: of a large vocabulary, less
    than the unfused program's. Its losses the unfused program's, its
    gradients too but where a bfloat16 rounds apart."""
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((n, d)) / np.sqrt(d)).astype(np.float32)
    w = rng.standard_normal((vocab, d)).astype(np.float32)
    t = rng.integers(0, vocab, n).astype(np.int64)
    c = (rng.random(n) / n).astype(np.float32)

    def trained(f):
        """A training step reducing the losses of `f` as the case says."""

        def step(h, w, t, c):
            """The reduced loss and the gradients of `h` and `w`."""
            losses = f(h, w, t)
            of = {
                "mean": lambda: F.mean(losses),
                "weighted": lambda: F.sum(losses * c),
                "squared": lambda: F.mean(losses) * F.mean(losses),
            }
            loss = of[reduce]()
            loss.backward()
            return loss, h.grad, w.grad

        return step

    def unfused(h, w, t):
        """The cross entropy of the matmul's logits, written without the fused path."""
        w = w.bfloat16() if head == "float32" else w
        return F.cross_entropy(F.matmul(h, w.t(), "float32", "float32"), t)

    try:
        H, W, T, C = (lumen.from_numpy(a).to(device) for a in (h, w, t, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    H, W = H.to(dtype="bfloat16"), W.to(dtype=head)
    step = trained(unfused)
    config = getattr(lumen.config.compiler, device)
    config.fused_linear_cross_entropy_chunk_size = chunk
    try:
        plan = lumen.graph_compiler.Plan(lumen.make_graph(step)(H, W, T, C), device)
        got = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(step)(H, W, T, C)]
    finally:
        config.fused_linear_cross_entropy_chunk_size = None
    sizes = [min(chunk, n - start) for start in range(0, n, chunk)]
    steps = plan.steps()
    labels = [s["label"] for s in steps]
    baseline = lumen.graph_compiler.Plan(lumen.make_graph(trained(unfused))(H, W, T, C), device)
    if len(sizes) == 1:

        def kernels(labels):
            """The kernels of `labels`, a horizontal fusion's each, but the chunks' fills and copies."""
            parts = [p for label in labels for p in label.split(" | ")]
            return sorted(p for p in parts if not re.fullmatch(r"(\d+x )?(full|dynamic_update_slice)", p))

        assert kernels(labels) == kernels(s["label"] for s in baseline.steps()), labels
    else:
        assert all(s["output"][2] != [n, vocab] for s in steps), labels
        assert ("linear_cross_entropy" in labels) == (reduce == "squared"), labels
        if reduce == "squared":
            sizes = []
        logits = [s["output"][2][0] for s in steps if s["label"] == "dot_general" and s["output"][2][1:] == [vocab]]
        rows = [
            s
            for s in steps
            if s["output"][2][1:] == [vocab] and [o[2] for o in s.get("extra_outputs", [])] == [s["output"][2][:1]]
        ]
        # On MPS a row this narrow runs in its matmul's kernel.
        one = device == "mps" and vocab <= 64
        assert sorted(logits) == ([] if one else sorted(sizes)), labels
        assert sorted(s["output"][2][0] for s in rows) == sorted(sizes), labels
        assert all(("dot_general" in s["label"]) == one for s in rows), labels
        if vocab > d:
            assert plan.workspace_bytes < baseline.workspace_bytes / 2, (
                plan.workspace_bytes,
                baseline.workspace_bytes,
            )
    want = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(trained(unfused))(H, W, T, C)]
    np.testing.assert_allclose(got[0], want[0], rtol=1e-6)
    for g, e in zip(got[1:], want[1:]):
        np.testing.assert_allclose(g, e, rtol=2.0**-7, atol=2.0**-7 * np.abs(e).max())
