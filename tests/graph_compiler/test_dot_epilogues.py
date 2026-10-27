"""Tests of matmul epilogues: the elementwise work and normalizations after a matmul fused into its kernel."""

import numpy as np
import pytest
from helpers import CUDA, MPS, meta

import lumen
import lumen.functional as F
from lumen import prims


class _ScaledMatmul(lumen.nn.Module):
    """Matmul by `w`, scaled by the runtime scalar `s`."""

    w: lumen.Tensor
    s: float

    def __call__(self, x):
        """`x @ w * s`."""
        return F.matmul(x, self.w, "float32", x.dtype) * self.s


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "f, shapes, dtype",
    [
        (lambda x, w, y: F.relu(F.matmul(x, w, "float32", x.dtype) + y[0]), [(64, 96), (96, 80), (2, 80)], "float32"),
        (lambda x, w, y: F.matmul(x, w, "float32", x.dtype) + y, [(3, 40, 24), (3, 24, 56), (3, 40, 56)], "float32"),
        (lambda x, w, y: F.matmul(x, w, "float32", x.dtype) + y, [(3, 40, 24), (24, 56), (3, 40, 56)], "float32"),
        (
            lambda x, w, y: F.relu(F.matmul(x, w, "float32", "float32")).bfloat16(),
            [(64, 96), (96, 80), (64, 80)],
            "bfloat16",
        ),
        (lambda x, w, y: F.matmul(x, w, "float32", x.dtype) * F.exp(y), [(64, 32), (32, 48), (64, 48)], "float16"),
        (
            lambda x, w, y: _silu(F.matmul(x, w, "float32", "float32")).to(dtype=x.dtype),
            [(64, 96), (96, 80), (64, 80)],
            "bfloat16",
        ),
        (
            lambda x, w, y: (F.matmul(x, w, "float32", "float32") * 0.5 + y.t()).bfloat16(),
            [(64, 96), (96, 80), (80, 64)],
            "float32",
        ),
        (
            lambda x, w, y: F.matmul(x, w, "float32", "float32").reshape(8, 8, 80) - y,
            [(64, 96), (96, 80), (80,)],
            "float32",
        ),
        (lambda x, w, y: F.tanh(F.matmul(x, w, "float32", "float32") + y), [(2, 32, 96), (96, 80), (80,)], "float32"),
    ],
    ids=[
        "bias relu",
        "batched residual",
        "folded residual",
        "relu cast",
        "times exp(y)",
        "silu in float32",
        "transposed residual",
        "reshaped",
        "batched by a matrix",
    ],
)
def test_matmul_epilogues_fuse(device, f, shapes, dtype, triton_on_cuda):
    """The elementwise primitives after a matmul (a bias, a residual, an
    activation, a cast, a product with another fused value) run in its
    kernel as it writes each output (``contraction_epilogues``): one step,
    exactly the unfused plan's result. On CUDA (Triton), one step too, the
    CPU's within rounding; float16 math has no Triton kernel."""
    rng = np.random.default_rng(0)
    try:
        arrays = [rng.standard_normal(s).astype(np.float32) for s in shapes]
        args = [lumen.from_numpy(a).to(device).to(dtype=dtype) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "cuda":
        if dtype == "float16":
            with pytest.raises(RuntimeError, match="convert to float32"):
                lumen.compile(f)(*args)
            return
        steps = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()
        assert len(steps) == 1, [s["label"] for s in steps]
        got = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
        cpu = [lumen.from_numpy(a).to(dtype=dtype) for a in arrays]
        want = lumen.to_numpy(lumen.compile(f, device="cpu")(*cpu).to(dtype="float32"))
        np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
        return
    (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps").steps()
    assert step["label"].startswith("dot_general →"), step["label"]
    fused = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        assert len(lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps").steps()) == 2
        unfused = lumen.to_numpy(lumen.compile(f)(*args).to(dtype="float32"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_array_equal(fused, unfused)


def _rms_block(x, wo, r, norm, w1):
    """A transformer block's residual, RMS norm and next projection: the
    residual stream ``x`` returned too, as a block reads it again."""
    x = r + F.matmul(x.bfloat16(), wo.t().bfloat16(), "float32", "float32")
    h = F.rms_norm(x, x.shape[-1], norm).bfloat16()
    return F.matmul(h, w1.t().bfloat16(), "float32", "float32"), x


def _layer_norm_block(x, wo, r, norm, w1):
    """A residual projection, a layer norm, then the next projection."""
    x = r + F.matmul(x.bfloat16(), wo.t().bfloat16(), "float32", "float32")
    c = x - F.mean(x, -1, keepdim=True)
    h = (c / F.sqrt(F.mean(c * c, -1, keepdim=True) + 1e-5) * norm).bfloat16()
    return F.matmul(h, w1.t().bfloat16(), "float32", "float32")


def _softmax_block(x, wo, r, norm, w1):
    """A residual projection, a scaled softmax, then the next projection."""
    x = r + F.matmul(x.bfloat16(), wo.t().bfloat16(), "float32", "float32")
    return F.matmul(F.softmax(x * norm, dim=-1).bfloat16(), w1.t().bfloat16(), "float32", "float32")


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("f", [_rms_block, _layer_norm_block, _softmax_block], ids=["rms", "layer", "softmax"])
@pytest.mark.parametrize("m, d", [(1024, 64), (1000, 48), (37, 64), (512, 128), (300, 200), (1000, 1000), (256, 4096)])
def test_matmul_epilogue_normalizes_its_rows(device, f, m, d, triton_on_cuda):
    """A normalization of a matmul's rows (a residual added first): rows no
    wider than its 64x64 tiles, the matmul's kernel normalizes each tile's
    rows from its accumulators (CODA's tile reductions), one kernel, the
    product never in memory, its residual stream written too where it is
    read again; exactly the unfused plan's result. On CUDA (Triton) the
    matmul's tiles hold whole rows up to 256 wide: one kernel normalizing
    them from its accumulators; wider (a multiple of 32), the matmul writes
    its tiles' partials of the rows' first reductions too, which the row
    kernel then reads; a softmax's (online) and other widths, a matmul then
    a row kernel. Wider rows: the matmul's
    kernel writes its product and each tile's partials of the rows' first
    reductions (an RMS norm's sum of squares, a layer norm's mean), then a
    row kernel takes them from those partials (CODA's following
    reduction), one step: the unfused result but where a row's sums add in
    another order. A softmax's max and sum (combined online) not so: a row
    kernel of their own, as before."""
    rng = np.random.default_rng(0)
    shapes = [(m, d), (d, d), (m, d), (d,), (96, d)]
    try:
        arrays = [rng.standard_normal(s).astype(np.float32) / 4 for s in shapes]
        args = [lumen.from_numpy(a).to(device) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    flatten = lambda out: [lumen.to_numpy(t) for t in (out if isinstance(out, tuple) else [out])]  # noqa: E731
    if device == "cuda":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "cuda").steps()
        rows = [s for s in steps if s["label"].startswith("dot_general") and "reduce" in s["label"]]
        assert len(rows) == (d <= 256), [s["label"] for s in steps]
        partials = [s for s in steps if s["label"].endswith("partials")]
        wide = d > 256 and d % 32 == 0 and f is not _softmax_block
        assert len(partials) == wide, [s["label"] for s in steps]
        got = flatten(lumen.compile(f)(*args))
        want = flatten(lumen.compile(f, device="cpu")(*(lumen.from_numpy(a) for a in arrays)))
        for g, w in zip(got, want):
            np.testing.assert_allclose(g, w, rtol=2e-2, atol=2e-2 * np.abs(w).max())
        return
    steps = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps").steps()
    rows = [s for s in steps if s["label"].startswith("dot_general") and "reduce" in s["label"]]
    wide = d > 64
    assert len(rows) == (not wide or f is not _softmax_block), [s["label"] for s in steps]
    fused = flatten(lumen.compile(f)(*args))
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        unfused = flatten(lumen.compile(f)(*args))
    finally:
        lumen.config.compiler.reset()
    for k, (got, want) in enumerate(zip(fused, unfused)):
        if wide and k == 0:
            np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2 * np.abs(want).max())
        else:
            np.testing.assert_array_equal(got, want)


class _NormedBlock(lumen.nn.Module):
    """A projection, its residual (the stream ``x``, read again after the
    norm: the next residual, the backward), an RMS norm of the stream as
    bfloat16 rounds it, and the next projection
    (``examples/transformer.py``'s)."""

    wo: lumen.Tensor
    norm: lumen.Tensor
    w1: lumen.Tensor

    def __call__(self, a, r):
        """Adds the projection to the stream, normalizes it and adds the next projection."""
        x = r + F.matmul(a.bfloat16(), self.wo.t().bfloat16(), "float32", "float32")
        h = F.rms_norm(x.bfloat16().float(), x.shape[-1], self.norm).bfloat16()
        return x + F.matmul(h, self.w1.t().bfloat16(), "float32", "float32")


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_trained_matmul_epilogue_normalizes_its_rows(device, triton_on_cuda):
    """Trained, a projection's residual and RMS norm one kernel still: its
    residual stream and the rounded copy the backward reads written by it
    too, read after it. Its gradients exactly the unfused plan's; on CUDA
    (Triton), the CPU's within rounding."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d = 1000, 64
    rng = np.random.default_rng(0)
    values = {
        "wo": rng.standard_normal((d, d)) / 8,
        "norm": 1 + rng.standard_normal(d) / 4,
        "w1": rng.standard_normal((d, d)) / 8,
    }
    a_r = [rng.standard_normal((m, d)).astype(np.float32) for _ in range(2)]
    a, r = (lumen.from_numpy(v).to(device) for v in a_r)

    def step(model, a, r):
        """Gradients of the summed squared output for the three weights."""
        y = model(a, r)
        F.sum(y * y).backward()
        return model.wo.grad, model.norm.grad, model.w1.grad

    def run(target=device):
        """The step's gradients compiled for `target`, on the test's values."""
        model = _NormedBlock(meta(d, d), meta(d), meta(d, d))
        f = lumen.compile(step, device=target)
        f(model, meta(m, d), meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v.astype(np.float32)) for k, v in values.items()})
        inputs = (a, r) if target == device else (lumen.from_numpy(v) for v in a_r)
        return [lumen.to_numpy(g) for g in f(model, *inputs)]

    if device == "cuda":
        for g, w in zip(run(), run("cpu")):
            assert np.linalg.norm(g - w) / np.linalg.norm(w) < 2e-2
        return

    graph = lumen.make_graph(step)(_NormedBlock(meta(d, d), meta(d), meta(d, d)), meta(m, d), meta(m, d))
    labels = [s["label"] for s in lumen.graph_compiler.Plan(graph, "mps").steps()]
    fused = [label for label in labels if label.startswith("dot_general → add") and "reduce_sum" in label]
    assert len(fused) == 1, labels
    got = run()
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        want = run()
    finally:
        lumen.config.compiler.reset()
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)


def _silu(h):
    """silu of `h`."""
    return h * F.sigmoid(h)


@pytest.mark.mps
def test_matmul_epilogue_stops_at_an_upcast_of_its_rounded_result():
    """A matmul whose result is rounded narrower than it accumulates
    (bfloat16 of float32) keeps a widening cast of it out of its epilogue:
    the epilogue fuses up to it (an add), the cast and what reads it a
    kernel after. Its result wanted wider, its output_dtype says so (one
    kernel). So too a float32 result the epilogue rounds (``.bfloat16()``,
    a backward's rounded gradient) and widens again for a reader that
    widens it itself: the kernel writes the bfloat16 value; widened for
    an output, the cast is the epilogue's (no kernel of its own)."""
    m = lambda *s, d="bfloat16": lumen.empty(list(s), dtype=d, device="meta")  # noqa: E731
    args = (m(64, 96), m(96, 80), m(64, 80, d="float32"))
    labels = lambda f: [
        s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps").steps()
    ]  # noqa: E731
    with pytest.warns(UserWarning, match="casts it back"):
        assert labels(lambda x, w, r: F.matmul(x, w, "float32", x.dtype).float() + r) == [
            "dot_general → cast(float32 -> bfloat16)",
            "cast(bfloat16 -> float32) → add",
        ]
    with pytest.warns(UserWarning, match="casts it back"):
        assert labels(lambda x, w, r: (F.matmul(x, w, "float32", x.dtype) + 1.0).float() + r) == [
            "dot_general → cast(float32 -> bfloat16) → add",
            "cast(bfloat16 -> float32) → add",
        ]
    assert labels(lambda x, w, r: F.matmul(x, w, "float32", "float32") + r) == ["dot_general → add"]
    rounded = lambda x, w: F.matmul(x, w, "float32", "float32").bfloat16().float()  # noqa: E731
    assert labels(lambda x, w, r: rounded(x, w) * r) == [
        "dot_general → cast(float32 -> bfloat16)",
        "cast(bfloat16 -> float32) → mul",
    ]
    assert labels(lambda x, w, r: rounded(x, w)) == [
        "dot_general → cast(float32 -> bfloat16) → cast(bfloat16 -> float32)"
    ]


def _dot(step):
    """A plan step's matmul: a ``dot_general`` step's text, or the dot of a
    fusion of the dot alone (the MPS compiler runs every float matmul as a
    GEMM fusion, epilogue or not); None for any other step."""
    if step["fusion"] is None:
        return step["text"] if step["text"].startswith("dot_general") else None
    nodes = [line.split(" = ", 1)[1] for line in step["fusion"]["body"].splitlines() if " = " in line]
    return nodes[0] if len(nodes) == 1 and nodes[0].startswith("dot_general") else None


@pytest.mark.mps
def test_narrowing_cast_of_a_matmul_is_its_output_dtype():
    """A matmul of bfloat16 operands writing its float32 accumulator as it
    is, read only rounded to bfloat16: the matmul writes bfloat16 (one
    kernel, epilogues or not), the same bits. Read in float32 too, it
    writes float32; rounded to another dtype (float16), the cast stays."""
    rng = np.random.default_rng(0)
    try:
        x, w = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((64, 96), (96, 80))
        )
    except RuntimeError as e:
        pytest.skip(str(e))

    def rounded(x, w):
        """The float32 matmul rounded to bfloat16."""
        y = F.matmul(x, w, "float32", "float32")
        return y.bfloat16()

    both = lambda x, w: (rounded(x, w), F.matmul(x, w, "float32", "float32"))  # noqa: E731

    def dots(f, *args):
        """Each MPS plan step of `f`'s graph as its matmul description, or its text."""
        return [_dot(s) or s["text"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps").steps()]

    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        (dot,) = dots(rounded, x, w)
        assert dot.startswith("dot_general") and "output_dtype=bf16" in dot
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(rounded)(x, w), "mps").steps()
        code = [open(f).read().splitlines()[line - 1].strip() for f, line in step["sources"]]
        assert sorted(code) == ["return y.bfloat16()", 'y = F.matmul(x, w, "float32", "float32")'], code
        assert [t.split("[")[0] for t in dots(both, x, w)] == ["dot_general", "cast(float32 -> bfloat16)"]
    finally:
        lumen.config.compiler.reset()
    got = lumen.to_numpy(lumen.compile(rounded)(x, w).to(dtype="float32"))
    want = lumen.to_numpy(lumen.compile(both)(x, w)[0].to(dtype="float32"))
    np.testing.assert_array_equal(got, want)
    half = lambda x, w: F.matmul(x, w, "float32", "float32").half()  # noqa: E731
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        assert [t.split("[")[0] for t in dots(half, x, w)] == ["dot_general", "cast(float32 -> float16)"]
    finally:
        lumen.config.compiler.reset()


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_matmul_epilogue_reads_the_rounded_output(device, triton_on_cuda):
    """A fused epilogue reads the matmul's output as its dtype rounds it
    (bf16, then cast up for a silu in float32), as traced; silu of the
    float32 accumulator is a dot_general with output_dtype float32."""
    rng = np.random.default_rng(0)
    try:
        x, w = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to(device).to(dtype="bfloat16")
            for s in ((64, 96), (96, 80))
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    dims = (((1,), (0,)), ((), ()))
    rounded = lumen.compile(lambda x, w: _silu(F.matmul(x, w, "float32", x.dtype).float()).to(dtype=x.dtype))(x, w)
    accum = lumen.compile(lambda x, w: _silu(prims.dot_general(x, w, dims, "float32", "float32")).to(dtype=x.dtype))(
        x, w
    )
    h = lumen.to_numpy(lumen.compile(lambda x, w: F.matmul(x, w, "float32", x.dtype).float())(x, w)).astype(np.float64)
    rounded, accum = (lumen.to_numpy(t.to(dtype="float32")) for t in (rounded, accum))
    np.testing.assert_allclose(rounded, h / (1 + np.exp(-h)), rtol=1e-2, atol=1e-2)
    assert not np.array_equal(rounded, accum)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_matmul_epilogue_takes_runtime_scalars_by_value(device, triton_on_cuda):
    """A module's float in a matmul's epilogue is a kernel argument: a new
    value runs the same kernel."""
    rng = np.random.default_rng(0)
    w, x = rng.standard_normal((32, 48)).astype(np.float32), rng.standard_normal((16, 32)).astype(np.float32)
    weight = lumen.empty([32, 48], device="meta")
    f = lumen.compile(lambda m, a: m(a), device=device)
    try:
        f(_ScaledMatmul(weight, 0.5), lumen.empty([16, 32], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    _ScaledMatmul(weight, 0.5).load_state_dict({"w": lumen.from_numpy(w)})
    for s in (0.5, 3.0):
        out = lumen.to_numpy(f(_ScaledMatmul(weight, s), lumen.from_numpy(x)))
        np.testing.assert_allclose(out, (x @ w) * s, rtol=1e-5, atol=1e-5)
    if device != "mps":
        return
    graph = lumen.make_graph(lambda m, a: m(a))(_ScaledMatmul(weight, 0.5), lumen.empty([16, 32], device="meta"))
    (step,) = lumen.graph_compiler.Plan(graph, "mps", parameters=[2], scalars=[1]).steps()
    assert ("s1", "float32", []) in step["inputs"] and "constant float &in" in step["fusion"]["source"]


class _Gated(lumen.nn.Module):
    """Gated MLP projection `relu(x @ w1) * (x @ w3)`."""

    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        """Applies the gated projection to `x`."""
        return F.relu(F.matmul(x, self.w1, "float32", x.dtype)) * F.matmul(x, self.w3, "float32", x.dtype)


class _Relu(lumen.nn.Module):
    """relu of a matmul."""

    w1: lumen.Tensor

    def __call__(self, x):
        """`relu(x @ w1)`."""
        return F.relu(F.matmul(x, self.w1, "float32", x.dtype))


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_matmul_epilogue_reads_packed_weights_in_place(device, triton_on_cuda):
    """A weight placed side by side with another (where dots merge) is a
    strided view; a matmul with its epilogue reads it in place, as the
    matmul alone does: no copy each call."""
    from lumen.profiler import ProfilerActivity, profile

    w1, w3 = lumen.empty([32, 48], device="meta"), lumen.empty([32, 48], device="meta")
    gated, relu = lumen.compile(lambda m, x: m(x), device=device), lumen.compile(lambda m, x: m(x), device=device)
    try:
        gated(_Gated(w1, w3), lumen.empty([16, 32], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    relu(_Relu(w1), lumen.empty([16, 32], device="meta"))
    rng = np.random.default_rng(0)
    w = rng.standard_normal((32, 48)).astype(np.float32)
    _Gated(w1, w3).load_state_dict({"w1": lumen.from_numpy(w), "w3": lumen.from_numpy(w)})
    x = rng.standard_normal((16, 32)).astype(np.float32)
    xt = lumen.from_numpy(x).to(device)
    np.testing.assert_allclose(lumen.to_numpy(relu(_Relu(w1), xt)), np.maximum(x @ w, 0), rtol=1e-5, atol=1e-5)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        relu(_Relu(w1), xt)
    ops = [e for e in prof.events() if e["kind"] == "op"]
    assert any(e["name"].startswith("dot_general →") for e in ops)
    copied = [e["inputs"] for e in ops if e["name"] == "lumen::contiguous"]
    assert [("float32", [32, 48])] not in copied, copied


class _ReluMLP(lumen.nn.Module):
    """Two-layer relu MLP."""

    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        """`relu(x @ w1) @ w2`."""
        return F.matmul(F.relu(F.matmul(x, self.w1, "float32", x.dtype)), self.w2, "float32", x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_matmul_epilogue_writes_values_the_backward_reads(device, triton_on_cuda):
    """In a training step relu's backward reads x @ w1: the relu still runs
    in the matmul's kernel, which writes x @ w1 too (XLA's GELU_AUX); the
    step agrees exactly with the unfused one. On CUDA (Triton), the CPU's
    within rounding."""
    rng = np.random.default_rng(0)
    w1, w2 = (rng.standard_normal(s).astype(np.float32) * 0.1 for s in ((64, 96), (96, 32)))
    x = rng.standard_normal((48, 64)).astype(np.float32)

    def step(m, x):
        """The loss and every parameter's gradient."""
        loss = F.sum(F.tanh(m(x)))
        loss.backward()
        return loss, [p.grad for p in m.parameters()]

    if device == "cuda":
        results = []
        for target in (device, "cpu"):
            model = _ReluMLP(lumen.empty([64, 96], device="meta"), lumen.empty([96, 32], device="meta"))
            f = lumen.compile(step, device=target)
            try:
                f(model, lumen.empty([48, 64], device="meta"))
            except RuntimeError as e:
                pytest.skip(str(e))
            model.load_state_dict({"w1": lumen.from_numpy(w1), "w2": lumen.from_numpy(w2)})
            loss, grads = f(model, lumen.from_numpy(x).to(target))
            results.append([lumen.to_numpy(t) for t in (loss, *grads)])
        for a, b in zip(*results):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4 * np.abs(b).max())
        return
    results = []
    for epilogues in (True, False):
        lumen.config.compiler.mps.contraction_epilogues = epilogues
        try:
            model = _ReluMLP(lumen.empty([64, 96], device="meta"), lumen.empty([96, 32], device="meta"))
            f = lumen.compile(step, device="mps")
            try:
                f(model, lumen.empty([48, 64], device="meta"))
            except RuntimeError as e:
                pytest.skip(str(e))
            model.load_state_dict({"w1": lumen.from_numpy(w1), "w2": lumen.from_numpy(w2)})
            loss, grads = f(model, lumen.from_numpy(x).to("mps"))
            results.append([lumen.to_numpy(t) for t in (loss, *grads)])
            plan = lumen.graph_compiler.Plan(lumen.make_graph(step)(model, lumen.from_numpy(x)), "mps")
            labels = [s["label"] for s in plan.steps()]
            if epilogues:
                assert "dot_general → max" in labels, labels
        finally:
            lumen.config.compiler.reset()
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
