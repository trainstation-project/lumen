"""Tests of normalization rows: softmax, RMS norm and layer norm forward and backward as one row kernel."""

import numpy as np
import pytest
from helpers import CUDA, MPS, rand

import lumen
import lumen.functional as F


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n", [16, 64])
def test_short_rows_reduce_several_a_threadgroup(device, n, triton_on_cuda):
    """Rows too short for a threadgroup each (an attention head's 16, a
    row of 64) are reduced as grouped lanes, several rows a threadgroup:
    the CPU's sums, the same each run."""
    x = rand(4096, n)
    try:
        X = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lambda x: F.sum(F.tanh(x * 2.0 + 1.0), -1)  # noqa: E731
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(X), "mps").steps()
        assert "reduce_grouped" in step["fusion"]["source"]
    got = [lumen.to_numpy(lumen.compile(f)(X)) for _ in range(2)]
    np.testing.assert_array_equal(got[0], got[1])
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(lumen.from_numpy(x)))
    np.testing.assert_allclose(got[0], want, rtol=1e-5, atol=1e-5)


class UpcastNorm(lumen.nn.Module):
    """Llama's RMS norm: normalized in float32, scaled in the input's dtype."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` in float32 and scales it in `x`'s dtype."""
        h = x.float()
        h = h / F.sqrt(F.mean(h * h, -1, keepdim=True) + self.eps)
        return self.weight * h.to(dtype=x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n", [300, 1024, 4096])
def test_upcast_rms_norm_is_one_kernel(device, n, triton_on_cuda):
    """An RMS norm in float32 of a bfloat16 input, cast back and scaled by
    its weight, is one row kernel: the cast and the weight run in its last
    pass (an epilogue). On MPS rows of up to 8 elements a thread read x
    once, keeping x.float() (its traced dtype) in registers for the last
    pass. It agrees with the CPU to a bfloat16 rounding."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    weight = lumen.empty([n], dtype="bfloat16", device="meta")
    graph = lumen.make_graph(lambda m, a: m(a))(
        UpcastNorm(weight, 1e-6), lumen.empty([8, n], dtype="bfloat16", device="meta")
    )
    (step,) = lumen.graph_compiler.Plan(graph, device, parameters=[2], scalars=[1]).steps()
    assert step["label"].endswith("div → cast(float32 -> bfloat16) → broadcast_in_dim → mul"), step["label"]
    if device == "mps":
        source = step["fusion"]["source"]
        cached = n <= 8 * 256
        assert ("float kept0[" in source) == cached
        assert source.count("_load0(in0, j)") == (1 if cached else 2), source
    x, w = rand(8, n), rand(n, seed=1)
    out = {}
    for target in ("cpu", device):
        f = lumen.compile(lambda m, a: m(a), device=target)
        norm = UpcastNorm(weight, 1e-6)
        f(norm, lumen.empty([8, n], dtype="bfloat16", device="meta"))
        norm.load_state_dict({"weight": lumen.from_numpy(w).to(dtype="bfloat16")})
        y = f(norm, lumen.from_numpy(x).to(dtype="bfloat16"))
        out[target] = lumen.to_numpy(y.to(dtype="float32"))
    np.testing.assert_allclose(out[device], out["cpu"], rtol=2**-7, atol=2**-7)


def _layer_norm(a, w):
    """Layer norm of `a` over its last dim, scaled by `w`."""
    xc = a - F.mean(a, -1, keepdim=True)
    return xc / F.sqrt(F.mean(xc * xc, -1, keepdim=True) + 1e-5) * w


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "f",
    [
        lambda a, w: a / F.sqrt(F.mean(a * a, -1, keepdim=True) + 1e-6) * w,
        lambda a, w: a / F.sqrt(F.sum(a * a, -1, keepdim=True) + 1e-6) * w,
        lambda a, w: a * (1.0 / F.sqrt(F.mean(a * a, -1, keepdim=True))),
        lambda a, w: (lambda e: e / F.sum(e, -1, keepdim=True))(F.exp(a - F.amax(a, -1, keepdim=True))),
        _layer_norm,
    ],
    ids=["rms mean", "rms sum", "rms reciprocal", "softmax written out", "layer norm"],
)
def test_normalizations_are_one_row_kernel(device, f, triton_on_cuda):
    """Whatever normalizes rows by reductions of them is a normalization
    diamond, or a chain of them (XLA's SoftmaxRewriterTriton), however it is
    written: one row kernel, its reductions inside, agreeing with the
    CPU. Nothing matches any one normalization. With no backward to read
    its values of a row (an RMS norm's sqrt(mean + eps)), it writes its
    output alone."""
    x, w = rand(64, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), device).steps()
    assert "reduce_" in step["label"], step["label"]
    assert step["extra_outputs"] == [], step["extra_outputs"]
    expected = lumen.to_numpy(lumen.compile(f, device="cpu")(lumen.from_numpy(x), lumen.from_numpy(w)))
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected, rtol=1e-5, atol=1e-5)


NORMALIZATIONS = [
    lambda a, w: a / F.sqrt(F.mean(a * a, -1, keepdim=True) + 1e-6) * w,
    lambda a, w: a / F.sqrt(F.sum(a * a, -1, keepdim=True) + 1e-6) * w,
    lambda a, w: a * (1.0 / F.sqrt(F.mean(a * a, -1, keepdim=True))),
    lambda a, w: (lambda e: e / F.sum(e, -1, keepdim=True))(F.exp(a - F.amax(a, -1, keepdim=True))),
    _layer_norm,
]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "f", NORMALIZATIONS, ids=["rms mean", "rms sum", "rms reciprocal", "softmax written out", "layer norm"]
)
def test_normalizations_train_in_one_row_kernel(device, f, triton_on_cuda):
    """Trained, a normalization's forward is still one row kernel: the
    values of a row its backward reads (an RMS norm's sqrt(mean + eps)) and
    its value before its epilogue (``y`` of ``y * w``, for w's gradient)
    are its outputs too, as a fused RMS norm writes each row's ``rstd``,
    never computed again; the gradients the CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(lambda a, w, c: F.sum(f(a, w) * c), (0, 1))
    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), device).steps()]
    if "sqrt" in str(labels):
        assert sum("sqrt" in label for label in labels) == 1, labels
    got = lumen.compile(grad)(X, W, C)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n", [12000, 100003], ids=["whole", "chunked"])
@pytest.mark.parametrize(
    "f", NORMALIZATIONS, ids=["rms mean", "rms sum", "rms reciprocal", "softmax written out", "layer norm"]
)
def test_wide_rows_are_one_row_kernel(device, n, f, triton_on_cuda):
    """Rows as wide as XMA's kernels take: on CUDA one row kernel, held
    whole up to 65536 columns, walked in chunks past that (a pass a
    reduction, softmax's max and sum in one online pass); its values and
    gradients the CPU's."""
    x, w, c = rand(4, n), rand(n, seed=1), rand(4, n, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "cuda":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), device).steps()
        assert "reduce_" in step["label"], step["label"]
    cpu = [lumen.from_numpy(a) for a in (x, w, c)]
    got = lumen.to_numpy(lumen.compile(f)(X, W).to("cpu"))
    np.testing.assert_allclose(got, lumen.to_numpy(lumen.compile(f, device="cpu")(*cpu[:2])), rtol=1e-4, atol=1e-6)
    grad = lumen.grad(lambda a, w, c: F.sum(f(a, w) * c), (0, 1))
    for g, e in zip(lumen.compile(grad)(X, W, C), lumen.compile(grad, device="cpu")(*cpu)):
        e = lumen.to_numpy(e)
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), e, rtol=1e-4, atol=1e-4 * np.abs(e).max())


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_rms_norm_backward_recomputes_its_normalized_input(device, triton_on_cuda):
    """A training step's RMS norm (its value read forward, its gradients
    taken): its forward row kernel writes its output and each row's
    ``sqrt(mean + eps)``, not ``x / rms`` (``x̂``, w's gradient's): the
    backward, reading ``x`` and ``rms`` anyway, divides again, a value of
    the rows' shape less kept from the forward. The loss and gradients the
    CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)

    def step(x, w, c):
        """The loss and the gradients of `x` and `w`."""
        loss = F.sum(F.rms_norm(x, 300, w) * c)
        loss.backward()
        return loss, x.grad, w.grad

    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    steps = lumen.graph_compiler.Plan(lumen.make_graph(step)(X, W, C), device).steps()
    (forward,) = [s for s in steps if "sqrt" in s["label"]]
    assert [o[2] for o in forward["extra_outputs"]] == [[64, 1]], forward["label"]
    got = lumen.compile(step)(X, W, C)
    want = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_wide_rms_norm_of_a_residual_kept_in_float32_fuses(device, triton_on_cuda):
    """A projection plus a residual kept in float32, its bfloat16 rounding
    RMS-normalized over rows too wide for one GEMM tile: on CUDA the GEMM
    writes both values and the norm's partials (``dot_partials``), as it
    writes a single one; the values the CPU's."""
    m, d = 64, 1024
    x, w, r, g = rand(m, d), rand(d, d, seed=1) / 32, rand(m, d, seed=2), rand(d, seed=3)

    def block(x, w, r, g):
        """The float32 residual sum and the RMS norm of its bfloat16 rounding."""
        s = F.matmul(x.bfloat16(), w.t().bfloat16(), "float32", "float32") + r
        f = s.bfloat16().float()
        return (f / F.sqrt(F.mean(f * f, -1, keepdim=True) + 1e-6) * g).bfloat16(), s

    try:
        args = [lumen.from_numpy(a).to(device) for a in (x, w, r, g)]
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "cuda":
        labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(block)(*args), device).steps()]
        assert any(label.endswith("partials") for label in labels), labels
    got = lumen.compile(block)(*args)
    want = lumen.compile(block, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, r, g)))
    for a, b in zip(got, want):
        a, b = (lumen.to_numpy(t.to("cpu").to(dtype="float32")) for t in (a, b))
        np.testing.assert_allclose(a, b, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_residual_read_by_an_rms_norm_and_the_loss_trains(device, triton_on_cuda):
    """A projection plus a residual, rounded to bfloat16, read by an RMS
    norm and by the loss: the norm's row writes its partials, so the fusion
    reading its root doesn't host that root too (a row root hosted
    elsewhere left its partials unwritten: the plan's steps waited on each
    other). The loss and gradients the CPU's."""
    x, w, r, g = rand(64, 64), rand(64, 64, seed=1) / 8, rand(64, 64, seed=2), rand(64, seed=3)

    def loss(x, w, r, g):
        """The summed norm and residual of the projection of `x`."""
        h = (F.matmul(x.bfloat16(), w.t().bfloat16(), "float32", "float32") + r).bfloat16()
        f = h.float()
        n = f / F.sqrt(F.mean(f * f, -1, keepdim=True) + 1e-6) * g
        return F.sum(n) + F.sum(f)

    step = lumen.value_and_grad(loss, (0, 1, 2, 3))
    try:
        args = [lumen.from_numpy(a).to(device) for a in (x, w, r, g)]
    except RuntimeError as e:
        pytest.skip(str(e))
    got = lumen.compile(step)(*args)
    want = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, r, g)))
    got, want = [got[0], *got[1]], [want[0], *want[1]]
    for a, b in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(a.to("cpu")), lumen.to_numpy(b), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "f, kernels",
    [
        (lambda x, w, c: F.sum(F.rms_norm(x, 300, w) * c), 3),
        (lambda x, w, c: F.sum(_layer_norm(x, w) * c), 4),
        (lambda x, w, c: F.sum(F.softmax(x * w, -1) * c), None),
    ],
    ids=["rms norm", "layer norm", "softmax"],
)
def test_normalization_backwards_are_row_kernels(device, f, kernels, triton_on_cuda):
    """A normalization's backward reduces each row and applies the result
    to the row's values, not one producer's (``dx = g·w / rms - x ·
    Σ_row(…)``): beyond a diamond, a row fusion still, one row kernel for
    x's gradient (a layer norm's two reductions in it), the weight's a sum
    over rows of its own; the forward's kernels besides. The gradients the
    CPU's."""
    x, w, c = rand(64, 300), rand(300, seed=1), rand(64, 300, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    if kernels is not None:
        steps = lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), device).steps()
        assert len(steps) == kernels, [s["label"] for s in steps]
    got = lumen.compile(grad)(X, W, C)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("n", [16, 64])
@pytest.mark.parametrize(
    "f",
    [
        lambda x, w, c: F.sum(F.rms_norm(x, x.shape[-1], w) * c),
        lambda x, w, c: F.sum((_layer_norm(x, w) + w) * c),
        lambda x, w, c: F.sum(F.softmax(x * w, -1) * c),
    ],
    ids=["rms norm", "layer norm", "softmax"],
)
def test_short_rows_are_a_simd_group_each(device, f, n, triton_on_cuda):
    """Row kernels of short rows (a model's width of 64, an attention
    head's 16) reduce each row in a SIMD group, eight a threadgroup (none
    left idle), forward and backward (its weight's gradient's blocks too,
    each SIMD group's rows combined in order): the CPU's values, the same
    bits each run; rows not a multiple of eight too."""
    x, w, c = rand(1003, n), rand(n, seed=1), rand(1003, n, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
        rows = [s for s in steps if s["fusion"] and "simd_sum" in s["fusion"]["source"]]
        assert rows, [s["label"] for s in steps]
    got = [[lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)] for _ in range(2)]
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, a, e in zip(*got, want):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_allclose(g, lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "f, sums",
    [
        (lambda x, w, c: F.sum(F.rms_norm(x, 64, w) * c), 1),
        (lambda x, w, c: F.sum((_layer_norm(x, w) + w) * c), 2),
    ],
    ids=["rms norm", "layer norm"],
)
def test_normalization_backwards_sum_weight_gradients_as_they_go(device, f, sums, triton_on_cuda):
    """A normalization's weight's gradient (``Σ_rows g·y``), a sum over
    rows its backward's row kernel reads the values of: that kernel adds
    them as it goes (a threadgroup or program a group of rows, each
    column's sum in registers), writing a group's sums each, then one
    kernel sums those (a group at a time, in order): two kernels for the
    backward, the sums the same bits every run, the gradients the CPU's."""
    x, w, c = rand(1024, 64), rand(64, seed=1), rand(1024, 64, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    steps = lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), device).steps()
    final = [s for s in steps if s["label"].startswith("reduce_sum") and s["inputs"][0][2] != [1024, 64]]
    assert len(final) == sums, [s["label"] for s in steps]
    groups = final[0]["inputs"][0][2]
    assert groups[1] == 64 and groups[0] < 1024 and all(s["inputs"][0][2] == groups for s in final)
    backward = [s for s in steps if any(o[2] == groups for o in s["extra_outputs"])]
    assert len(backward) == 1, [s["label"] for s in steps]
    assert sum(o[2] == groups for o in backward[0]["extra_outputs"]) == sums
    got = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    again = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, a, e in zip(got, again, want):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_allclose(g, lumen.to_numpy(e), rtol=1e-4, atol=1e-4)
