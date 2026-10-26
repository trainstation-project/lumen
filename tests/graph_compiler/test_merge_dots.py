"""Tests of merged dots: matmuls sharing an operand run as one over packed weights, forward and backward."""

import re
import warnings

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, meta, rand

import lumen
import lumen.functional as F
from lumen.profiler import ProfilerActivity, profile


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_packed_weights_are_one_matmul(device, triton_on_cuda):
    """relu(x @ w1) * (x @ w3) with w1 and w3 packed into one weight: one
    matmul, on MPS the gate its epilogue (its halves a gated pair)."""
    x, w13 = rand(16, 32), rand(32, 128, seed=1)
    try:
        args = [lumen.from_numpy(a).to(device) for a in (x, w13)]
    except RuntimeError as e:
        pytest.skip(str(e))

    def gated(x, w13):
        """relu of one half of the packed matmul times the other."""
        gate, up = F.matmul(x, w13, "float32", x.dtype).chunk(2, -1)
        return F.relu(gate) * up

    out = lumen.to_numpy(lumen.compile(gated)(*args))
    h = x @ w13
    np.testing.assert_allclose(out, np.maximum(h[:, :64], 0) * h[:, 64:], rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(gated)(*args)
    steps = [s["primitive"] for s in lumen.graph_compiler.Plan(graph, device).steps()]
    if device == "mps":
        assert steps == ["dot_general → max → mul"], steps
    elif device == "cuda":
        assert len(steps) == 1, steps
    else:
        assert steps.count("dot_general") == 1


class Gated(lumen.nn.Module):
    """Gated MLP projection `relu(x @ w1) * (x @ w3)`."""

    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        """Applies the gated projection to `x`."""
        return F.relu(F.matmul(x, self.w1, "float32", x.dtype)) * F.matmul(x, self.w3, "float32", x.dtype)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_dots_sharing_an_operand_merge(device, triton_on_cuda):
    """relu(x @ w1) * (x @ w3) with weights w1 and w3: on MPS, one matmul of
    x and a block holding w1 and w3 interleaved (XLA's DotMerger, with the
    weights placed together instead of concatenated; a gated pair's columns
    interleaved, w1's j its 2j, w3's its 2j + 1), the gate its epilogue (a
    gated pair: one kernel)."""
    x, w1v, w3v = rand(16, 32), rand(32, 64, seed=1), rand(32, 64, seed=2)
    model = Gated(meta(32, 64), meta(32, 64))
    f = lumen.compile(lambda model, x: model(x), device=device)
    try:
        f(model, meta(16, 32))
    except RuntimeError as e:
        pytest.skip(str(e))
    model.load_state_dict({"w1": lumen.from_numpy(w1v), "w3": lumen.from_numpy(w3v)})
    out = lumen.to_numpy(f(model, lumen.from_numpy(x)))
    np.testing.assert_allclose(out, np.maximum(x @ w1v, 0) * (x @ w3v), rtol=1e-5, atol=1e-5)
    graph = lumen.make_graph(lambda model, x: model(x))(model, lumen.from_numpy(x))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=[1, 2], packable=[1, 2])
    steps = [s["primitive"] for s in plan.steps()]
    if device == "mps":
        assert plan.packed == [([1, 2], 1)] and model.w1._placed("mps").shares_storage_with(model.w3._placed("mps"))
        assert plan.interleaved == [[2]]
        w1, w3 = model.w1._placed("mps"), model.w3._placed("mps")
        assert (list(w1.strides), list(w3.strides)) == ([128, 2], [128, 2])
        assert w3.storage_offset - w1.storage_offset == 1
        assert len(steps) == 1 and "concatenate" not in steps
        assert plan.steps()[0]["label"] == "2x dot_general → max → mul"
        from lumen.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            f(model, lumen.from_numpy(x))
            lumen.mps.synchronize()
        names = {(e["kind"], e["name"]) for e in prof.events()}
        assert {("op", "2x dot_general → max → mul"), ("gpu", "2x dot_general → max → mul")} <= names
    elif device == "cuda":
        assert plan.packed == [([1, 2], 1)] and model.w1._placed("cuda").shares_storage_with(model.w3._placed("cuda"))
        assert plan.interleaved == [[2]] and len(steps) == 1, steps
    else:
        assert plan.packed == [] and [s.split(" → ")[0] for s in steps].count("dot_general") == 2


class Attention(lumen.nn.Module):
    """Single-head softmax attention with q, k and v projections."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        """Attends `x` to itself."""
        q, k, v = (
            F.matmul(x, self.wq, "float32", x.dtype),
            F.matmul(x, self.wk, "float32", x.dtype),
            F.matmul(x, self.wv, "float32", x.dtype),
        )
        return F.matmul(F.softmax(F.matmul(q, k.t(), "float32", x.dtype), -1), v, "float32", x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_attention_projections_merge_into_one_matmul(device, triton_on_cuda):
    """q, k and v (x @ wq, x @ wk, x @ wv) as one matmul of x and a block
    of the three weights; the attention (one flash-attention kernel) reads
    them in place. As traced, the matmuls reading q, k and v read them in
    place, strided views of its result (k's transpose read as the dot's
    dimensions)."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    model = Attention(meta(32, 32), meta(32, 32), meta(32, 32))
    f = lumen.compile(lambda model, x: model(x), device=device)
    f(model, meta(16, 32))
    values = [rand(32, 32, seed=i + 1) / 4 for i in range(3)]
    model.load_state_dict({n: lumen.from_numpy(v) for n, v in zip(("wq", "wk", "wv"), values)})
    x = rand(16, 32)
    q, k, v = (x @ w for w in values)
    s = np.exp(q @ k.T - (q @ k.T).max(-1, keepdims=True))
    expected = (s / s.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(lumen.to_numpy(f(model, lumen.from_numpy(x))), expected, rtol=1e-4, atol=1e-4)
    if device != "mps":
        return
    graph = lumen.make_graph(lambda model, x: model(x))(model, lumen.from_numpy(x))
    plan = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
    merged, attention = plan.steps()
    assert (merged["label"], attention["label"]) == ("3x dot_general", "flash_attention")
    assert attention["inputs"] == [(merged["output"][0], "float32", [16, 96])]
    lumen.config.compiler.mps.flash_attention = False
    try:
        plan = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
    finally:
        lumen.config.compiler.reset()
    steps = plan.steps()
    assert plan.packed == [([1, 2, 3], 1)]
    dots = [s["label"] for s in steps if s["label"].startswith(("3x dot_general", "dot_general"))]
    assert len(dots) == 3 and dots[0] == "3x dot_general" and dots[2] == "dot_general", dots
    assert dots[1].startswith("dot_general → reduce_max") and dots[1].endswith("div"), dots
    assert "slice" not in [s["primitive"] for s in steps]
    views = [v for s in steps for v in s["views"] if v is not None]
    assert views == [(0, [96, 1]), (32, [96, 1]), (64, [96, 1])]


class Heads(lumen.nn.Module):
    """q, k and v projections of ``[batch * seq, dim]`` rows (``nn.Linear``'s
    ``[out, in]`` weights, merged in training too), attended over 4 heads by
    ``F.flash_attention``."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        """Causal flash attention of the projections over 4 heads of 16."""
        q, k, v = (F.matmul(x, w.t(), "float32", x.dtype).reshape(2, 32, 4, 16) for w in (self.wq, self.wk, self.wv))
        return F.flash_attention(q, k, v, is_causal=True).reshape(64, 64)


@pytest.mark.mps
def test_trained_attention_reads_merged_projections_in_place():
    """In training, the attention's forward and backward kernels both read
    q, k and v where the merged matmul writes them (each a strided view of
    its result, read through its slice and reshape): no copy of each,
    though two kernels read it."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    model = Heads(meta(64, 64), meta(64, 64), meta(64, 64))
    grad = lumen.grad(lambda model, x: F.sum(model(x) * model(x)), (0, 1))
    graph = lumen.make_graph(grad)(model, meta(64, 64))
    steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3]).steps()
    labels = [s["label"] for s in steps]
    merged = steps[labels.index("3x dot_general")]["output"][0]
    attention = [s for s in steps if s["label"].startswith("flash_attention")]
    assert len(attention) >= 2, labels
    for s in attention:
        assert merged in [b for b, *_ in s["inputs"]], (s["label"], s["inputs"])
    assert not [label for label in labels if label.startswith("slice")], labels
    assert not [label for label in labels if "concatenate" in label], labels


class QKV(lumen.nn.Module):
    """Three projections of x, ``nn.Linear``'s way: ``[out, in]`` weights
    read as ``x @ w.t()``."""

    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, x):
        """The three projections of `x`."""
        return [F.matmul(x, w.t(), "float32", x.dtype) for w in (self.wq, self.wk, self.wv)]


class MixedQKV(QKV):
    """:class:`QKV` in mixed precision, by hand: float32 weights cast to
    bfloat16 where the matmuls read them, the result float32."""

    def __call__(self, x):
        """The three projections in bfloat16 with a float32 result."""
        return [F.matmul(x.bfloat16(), w.t().bfloat16(), "float32", "float32") for w in (self.wq, self.wk, self.wv)]


def test_gradients_through_a_rounding_are_written_wide():
    """A float32 value rounded to bfloat16 for matmuls (mixed precision's
    weights, ``w.t().bfloat16()``; ``x.bfloat16()``, read by three): its
    gradient the matmuls' writing their float32 accumulator (summed for
    ``x``), not rounded to bfloat16 and cast back: no precision warning, no
    cast to float32. The gradients NumPy's, to bfloat16 operands' rounding."""
    n, d = 16, 32

    def step(model, x):
        """Gradients of the summed squares of the projections for `x` and each weight."""
        F.sum(sum(F.sum(y * y) for y in model(x))).backward()
        return x.grad, model.wq.grad, model.wk.grad, model.wv.grad

    model = MixedQKV(meta(d, d), meta(d, d), meta(d, d))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = lumen.make_graph(step)(model, meta(n, d))
    assert not any(node["text"] == "cast[new_dtype=f32]" for node in graph.nodes())
    values = {k: rand(d, d, seed=i) / 4 for i, k in enumerate(("wq", "wk", "wv"))}
    x = rand(n, d, seed=3)
    f = lumen.compile(step, device="cpu")
    model = MixedQKV(meta(d, d), meta(d, d), meta(d, d))
    f(model, meta(n, d))
    model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
    got = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    ws = [values[k].astype(np.float64) for k in ("wq", "wk", "wv")]
    ys = [x @ w.T for w in ws]
    want = [sum(2 * y @ w for y, w in zip(ys, ws))] + [2 * y.T @ x for y in ys]
    for g, w in zip(got, want):
        assert np.linalg.norm(g - w) / np.linalg.norm(w) < 2e-2


class SwiGLU(lumen.nn.Module):
    """A gated MLP's up projection, ``silu(x @ w1.t()) * (x @ w3.t())``: in
    float32, or (``mixed``) of bfloat16 operands, the gate in float32, the
    result bfloat16 (read back as float32)."""

    w1: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x, mixed=False, gate=False):
        """The gated value, plus the gate when `gate`; `mixed` uses bfloat16 operands."""
        if mixed:
            g = F.matmul(x.bfloat16(), self.w1.t().bfloat16(), "float32", "float32")
            u = F.matmul(x.bfloat16(), self.w3.t().bfloat16(), "float32", "float32")
            z = (g * F.sigmoid(g) * u).bfloat16().float()
        else:
            g, u = F.matmul(x, self.w1.t(), "float32", x.dtype), F.matmul(x, self.w3.t(), "float32", x.dtype)
            z = g * F.sigmoid(g) * u
        return (z, g) if gate else z


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("mixed", [False, True], ids=["float32", "mixed precision"])
@pytest.mark.parametrize("hidden", [256, 344], ids=["even tiles", "ragged"])
def test_gated_pair_is_its_merged_matmul_epilogue(device, mixed, hidden, triton_on_cuda):
    """Two projections of one input combined elementwise (SwiGLU's gate and
    value): one matmul of their block whose epilogue computes the gate (its
    tile holding each pair's columns side by side, ``GROUP`` 2), neither
    projection written out; the values the CPU's. One whose gate is an
    output too is paired as well, the epilogue writing the gate out
    beside the product; its values the CPU's. In mixed precision the
    input's and the block's casts to bfloat16 are one kernel before it."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d = 300, 128
    values = {"w1": rand(hidden, d) / np.sqrt(d), "w3": rand(hidden, d, seed=1) / np.sqrt(d)}
    x = rand(m, d, seed=2)
    for gate in (False, True):
        out = {}
        for target in ("cpu", device):
            model = SwiGLU(meta(hidden, d), meta(hidden, d))

            def f(model, x):
                """Runs the model with the case's `mixed` and `gate` flags."""
                return model(x, mixed, gate)

            g = lumen.compile(f, device=target)
            g(model, meta(m, d))
            model.load_state_dict({k: lumen.from_numpy(v.astype(np.float32)) for k, v in values.items()})
            result = g(model, lumen.from_numpy(x))
            out[target] = [lumen.to_numpy(v) for v in (result if gate else [result])]
            if target == "mps":
                graph = lumen.make_graph(f)(model, meta(m, d))
                steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2], packable=[1, 2]).steps()
                dots = [s["label"] for s in steps if "dot_general" in s["label"]]
                assert len(dots) == 1 and dots[0].startswith("2x dot_general → logistic → mul → mul"), dots
                if not gate:
                    assert len(steps) == (2 if mixed else 1), [s["label"] for s in steps]
                    (merged,) = [s for s in steps if s["label"].startswith("2x dot_general")]
                    code = open(__file__).read().splitlines()
                    traced = "\n".join(code[line - 1] for file, line in merged["sources"] if file == __file__)
                    assert "self.w1.t()" in traced and "self.w3.t()" in traced, merged["sources"]
        tol = 2e-2 if mixed else 1e-5
        for got, want in zip(out[device], out["cpu"]):
            np.testing.assert_allclose(got, want, atol=tol * np.abs(want).max())


class GatedMLP(lumen.nn.Module):
    """A projection, then a gated MLP of bfloat16 matmuls (ReGLU, or
    SwiGLU's gate in float32): the projection so its input needs a
    gradient, as a layer's does in a model."""

    w0: lumen.Tensor
    wg: lumen.Tensor
    wu: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, inp, swiglu):
        """Projects `inp`, then applies the ReGLU or SwiGLU MLP."""
        x = F.matmul(inp, self.w0.t().bfloat16(), "float32", "bfloat16")
        if swiglu:
            g = F.matmul(x, self.wg.t().bfloat16(), "float32", "float32")
            u = F.matmul(x, self.wu.t().bfloat16(), "float32", "float32")
            h = (g * F.sigmoid(g) * u).bfloat16()
        else:
            h = F.relu(F.matmul(x, self.wg.t().bfloat16(), "float32", "bfloat16")) * F.matmul(
                x, self.wu.t().bfloat16(), "float32", "bfloat16"
            )
        return F.matmul(h, self.w2.t().bfloat16(), "float32", "bfloat16")


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("swiglu", [False, True], ids=["reglu", "swiglu"])
def test_gated_pair_backward_merges_its_gradients_dots(device, swiglu, triton_on_cuda):
    """A gated pair's backward (``merge_dots_backward``): its input's gradient one
    dot of its cotangents and the forward's weight block, both interleaved
    (K twice as long, no copy of the weights), its weights' gradients one
    dot, its rows interleaved as the block is, each weight's written apart
    by its epilogue; the cotangents the down projection's
    gradient GEMM's expanding epilogue, written interleaved, whatever the
    gate (SwiGLU's in float32, from the rounded gradient widened: an upcast
    only that epilogue takes). The forward the merged dot's paired epilogue,
    its halves (which the backward reads) written by it too. The gradients
    the CPU's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def step(model, x):
        """Gradients of the summed squared output for every weight."""
        y = model(x.bfloat16(), swiglu).float()
        F.sum(y * y).backward()
        return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

    grads = {}
    for target in ("cpu", device):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step, device=target)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads[target] = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    for got, want in zip(grads[device], grads["cpu"]):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    if device != "mps":
        return
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step)(model, meta(m, d))
    steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    labels = [s["label"] for s in steps]
    merged = [s for s in steps if s["label"].startswith("2x dot_general")]
    assert sorted(s["output"][2] for s in merged) == sorted([[m, h], [m, d], [h, d]]), labels
    forward = [s for s in merged if s["output"][2] == [m, h]]
    assert len(forward) == 1 and "mul" in forward[0]["label"], labels
    assert not any(s["label"].startswith("slice") and s["output"][2] == [m, h] for s in steps), labels
    expanding = [s for s in steps if "dot_general" in s["label"] and s["label"].endswith("concatenate")]
    assert len(expanding) == 1 and expanding[0]["output"][2] == [m, h, 2], labels
    assert sum("concatenate" in label for label in labels) == 1, labels


class _GatedMLPFunction(lumen.autograd.Function):
    """``sigmoid(x wg^T) * (x wu^T)`` then ``wd``, in bfloat16 matmuls, its
    backward by hand (``examples/transformer.py``'s): the weights' gradients
    ``g.t() @ x``, transposes of computed values."""

    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        """The gated MLP's output, saving its inputs and pre-activations."""
        g = F.matmul(x, wg.t().bfloat16(), "float32", "float32")
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        ctx.save_for_backward(x, g, u, wg, wu, wd)
        return F.matmul((F.sigmoid(g) * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    @staticmethod
    def backward(ctx, dy):
        """Hand-written gradients of `x` and the three weights."""
        x, g, u, wg, wu, wd = ctx.saved_tensors
        s = F.sigmoid(g)
        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), (s * u).bfloat16(), "float32", "float32")
        du = (dh * s).bfloat16()
        dg = (dh * u * s * (1 - s)).bfloat16()
        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32") + F.matmul(du, wu.bfloat16(), "float32", "float32")
        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")
        return dx.bfloat16(), dwg, dwu, dwd


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_hand_written_merge_dots_backward_merges_as_autodiffs(device, triton_on_cuda):
    """A gated pair's backward written by hand (a ``Function``, its
    weights' gradients ``g.t() @ x``): merged as autodiff's is, the
    transposes read as the dots' dimensions (no copies): the cotangents
    the down projection's gradient GEMM's expanding epilogue, x's gradient
    one dot of the block, the weights' one dot writing each apart. The
    gradients autodiff's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def autodiff(x, wg, wu, wd):
        """The same gated MLP for autodiff to differentiate."""
        g = F.matmul(x, wg.t().bfloat16(), "float32", "float32")
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        return F.matmul((F.sigmoid(g) * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    def step(mlp):
        """A training step running `mlp` after an input projection."""

        def f(model, x):
            """Gradients of the summed squared output for every weight."""
            x = F.matmul(x.bfloat16(), model.w0.t().bfloat16(), "float32", "bfloat16")
            y = mlp(x, model.wg, model.wu, model.w2).float()
            F.sum(y * y).backward()
            return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

        return f

    grads = []
    for mlp in (_GatedMLPFunction.apply, autodiff):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step(mlp), device=device)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads.append([lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))])
    for got, want in zip(*grads):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    if device != "mps":
        return
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step(_GatedMLPFunction.apply))(model, meta(m, d))
    steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    labels = [s["label"] for s in steps]
    expanding = [s for s in steps if "dot_general" in s["label"] and s["label"].endswith("concatenate")]
    assert len(expanding) == 1 and expanding[0]["output"][2] == [m, h, 2], labels
    assert sum("concatenate" in label for label in labels) == 1, labels
    assert any(s["label"] == "2x dot_general" and s["output"][2] == [h, d] for s in steps), labels
    assert not any("transpose" in label for label in labels), labels


class ProjectedAttention(lumen.nn.Module):
    """A projection, then attention of its q, k and v (bfloat16 matmuls):
    the projection so its input needs a gradient."""

    w0: lumen.Tensor
    wq: lumen.Tensor
    wk: lumen.Tensor
    wv: lumen.Tensor

    def __call__(self, inp, batch, heads):
        """Projects `inp`, then attends its q, k and v over `heads` heads per batch."""
        x = F.matmul(inp, self.w0.t().bfloat16(), "float32", "bfloat16")
        tokens, dim = x.shape
        q, k, v = (
            F.matmul(x, w.t().bfloat16(), "float32", "bfloat16")
            .reshape(batch, tokens // batch, heads, dim // heads)
            .transpose(-2, -3)
            for w in (self.wq, self.wk, self.wv)
        )
        s = F.matmul(q, k.transpose(-1, -2), "float32", "float32")
        return F.matmul(F.softmax(s, dim=-1).bfloat16(), v, "float32", "float32")


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("deterministic", [False, True], ids=["atomic", "deterministic"])
def test_qkv_backward_merges_its_gradients_dots(device, deterministic, triton_on_cuda):
    """Attention's q, k and v projections' backward (``merge_dots_backward``, a
    group of three): its input's gradient one dot of ``[dq | dk | dv]``
    and the forward's weight block (K three times as long), its weights'
    gradients one dot. ``[dq | dk | dv]`` written in place by the flash
    attention backward's kernels, each gradient cast to bfloat16 and laid
    out as the dots read it as it is stored: no concatenate. Deterministic
    (dQ its own kernel), no kernel between them and the dots. Else (one
    kernel, adding dQ atomically in float32) dQ cast and laid out by a
    kernel of its own, into its place. The gradients the CPU's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    batch, seq, d, heads = 4, 32, 64, 4
    m = batch * seq
    values = {name: rand(d, d, seed=k) / 8 for k, name in enumerate(("w0", "wq", "wk", "wv"))}
    x = rand(m, d, seed=4)

    def step(model, x):
        """Gradients of the summed squared attention output for every weight."""
        y = model(x.bfloat16(), batch, heads)
        F.sum(y * y).backward()
        return model.w0.grad, model.wq.grad, model.wk.grad, model.wv.grad

    before = lumen.config.compiler.mps.deterministic
    lumen.config.compiler.mps.deterministic = deterministic
    try:
        grads = {}
        for target in ("cpu", device):
            model = ProjectedAttention(meta(d, d), meta(d, d), meta(d, d), meta(d, d))
            f = lumen.compile(step, device=target)
            f(model, meta(m, d))
            model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
            grads[target] = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
        if device == "mps":
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                f(model, lumen.from_numpy(x))
            model = ProjectedAttention(meta(d, d), meta(d, d), meta(d, d), meta(d, d))
            graph = lumen.make_graph(step)(model, meta(m, d))
            steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    finally:
        lumen.config.compiler.mps.deterministic = before
    for got, want in zip(grads[device], grads["cpu"]):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    if device != "mps":
        return
    labels = [s["label"] for s in steps]
    merged = [s for s in steps if s["label"].startswith("3x dot_general")]
    assert sorted(s["output"][2] for s in merged) == sorted([[m, 3 * d], [3 * d, d], [m, d]]), labels
    (dx,) = [s for s in merged if s["output"][2] == [m, d]]
    assert [i[2] for i in dx["inputs"]] == [[m, 3 * d], [3 * d, d]], labels
    assert "concatenate" not in labels, labels
    backward = [s for s in steps if s["label"].startswith("flash_attention_backward")]
    assert all(s["output"][1:] == ("bfloat16", [m, d]) for s in backward), labels
    written = [e for e in prof.events() if e["name"].startswith("flash_attention_backward")]
    if deterministic:
        assert len(backward) == 2, labels
        assert backward[0]["output"][0] == backward[1]["output"][0] == dx["inputs"][0][0], steps
        assert [e["output_strides"] for e in written] == [[[3 * d, 1]] * 2, [[3 * d, 1]]], written
        at = labels.index(backward[-1]["label"])
        assert all(label.startswith("3x dot_general") for label in labels[at + 1 : at + 3]), labels
        assert not any("concatenate" in label or "transpose" in label for label in labels), labels
    else:
        (step,) = backward
        assert step["output"][0] == dx["inputs"][0][0], steps
        (event,) = written
        assert event["output_strides"][:2] == [[3 * d, 1]] * 2, event
        assert event["outputs"][2][0] == "float32", event
        at = labels.index(step["label"])
        cast = steps[at + 1]
        assert "cast" in cast["label"] and cast["output"] == (dx["inputs"][0][0], "bfloat16", [m, d]), steps
        assert labels[at + 2].startswith("3x dot_general"), labels
        (event,) = [e for e in prof.events() if e["name"] == cast["label"] and e["inputs"][0][0] == "float32"]
        assert event["output_strides"] == [[3 * d, 1]], event


class _SavedNarrowMLP(lumen.autograd.Function):
    """A gated MLP saving its gate and up projections rounded to bfloat16
    (``examples/transformer.py``'s): its backward reads them so."""

    @staticmethod
    def forward(ctx, x, wg, wu, wd):
        """The gated MLP's output, saving the gate and up values rounded to bfloat16."""
        s = F.sigmoid(F.matmul(x, wg.t().bfloat16(), "float32", "float32"))
        u = F.matmul(x, wu.t().bfloat16(), "float32", "float32")
        ctx.save_for_backward(x, s.bfloat16(), u.bfloat16(), wg, wu, wd)
        return F.matmul((s * u).bfloat16(), wd.t().bfloat16(), "float32", "bfloat16")

    @staticmethod
    def backward(ctx, dy):
        """Hand-written gradients reading the saved bfloat16 values."""
        x, s, u, wg, wu, wd = ctx.saved_tensors
        s, u = s.float(), u.float()
        dh = F.matmul(dy, wd.bfloat16(), "float32", "float32")
        dwd = F.matmul(dy.t(), (s * u).bfloat16(), "float32", "float32")
        du = (dh * s).bfloat16()
        dg = (dh * u * s * (1 - s)).bfloat16()
        dx = F.matmul(dg, wg.bfloat16(), "float32", "float32") + F.matmul(du, wu.bfloat16(), "float32", "float32")
        dwg = F.matmul(dg.t(), x, "float32", "float32")
        dwu = F.matmul(du.t(), x, "float32", "float32")
        return dx.bfloat16(), dwg, dwu, dwd


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_values_saved_narrower_are_written_narrower(device, triton_on_cuda):
    """Values a matmul's epilogue computes that are read outside it only
    rounded narrower (a backward's saved bfloat16 halves of a gated pair):
    the epilogue writes the rounded values, as the program keeps them, not
    the float32 ones for the backward to round. The gradients the CPU's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    m, d, h = 384, 64, 256
    values = {
        "w0": rand(d, d) / 8,
        "wg": rand(h, d, seed=1) / 8,
        "wu": rand(h, d, seed=2) / 8,
        "w2": rand(d, h, seed=3) / 16,
    }
    x = rand(m, d, seed=4)

    def step(model, x):
        """Gradients through the saved-narrow MLP after an input projection."""
        x = F.matmul(x.bfloat16(), model.w0.t().bfloat16(), "float32", "bfloat16")
        y = _SavedNarrowMLP.apply(x, model.wg, model.wu, model.w2).float()
        F.sum(y * y).backward()
        return model.w0.grad, model.wg.grad, model.wu.grad, model.w2.grad

    grads = {}
    for target in ("cpu", device):
        model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
        f = lumen.compile(step, device=target)
        with pytest.warns(UserWarning, match="the rounding loses precision"):
            f(model, meta(m, d))
        model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
        grads[target] = [lumen.to_numpy(g) for g in f(model, lumen.from_numpy(x))]
    for got, want in zip(grads[device], grads["cpu"]):
        assert np.linalg.norm(got - want) / np.linalg.norm(want) < 2e-2
    if device != "mps":
        return
    model = GatedMLP(meta(d, d), meta(h, d), meta(h, d), meta(d, h))
    with pytest.warns(UserWarning, match="the rounding loses precision"):
        graph = lumen.make_graph(step)(model, meta(m, d))
    steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3, 4], packable=[1, 2, 3, 4]).steps()
    (forward,) = [s for s in steps if s["label"].startswith("2x dot_general") and "logistic" in s["label"]]
    assert [o[1:] for o in forward["extra_outputs"]] == [("bfloat16", [m, h])] * 2, forward["label"]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("module", [QKV, MixedQKV], ids=["float32", "mixed precision"])
def test_projections_of_several_inputs_each_read_one_block(device, module, triton_on_cuda):
    """The same weights projecting two inputs (a gradient accumulation's
    passes, each its own activations): one block of them, each input's
    three projections one matmul reading it (its chain, a bfloat16 copy,
    computed once), the values as unmerged."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(model, x1, x2):
        """relu of each projection of both inputs."""
        return [F.relu(v) for v in [*model(x1), *model(x2)]]

    if device == "mps":
        model = module(meta(64, 64), meta(64, 64), meta(64, 64))
        graph = lumen.make_graph(f)(model, meta(32, 64), meta(32, 64))
        steps = lumen.graph_compiler.Plan(graph, "mps", parameters=[2, 3, 4], packable=[2, 3, 4]).steps()
        labels = [s["label"] for s in steps]
        assert labels.count("3x dot_general") == 2, labels
        blocks = {i[0] for s in steps if s["label"] == "3x dot_general" for i in s["inputs"]}
        assert len(blocks) == 3, blocks
    values = rand(64, 64), rand(64, 64, seed=1), rand(64, 64, seed=2)
    xs = [lumen.from_numpy(rand(32, 64, seed=k)).to(device) for k in (3, 4)]
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = module(meta(64, 64), meta(64, 64), meta(64, 64))
            g = lumen.compile(f, device=device)
            g(model, meta(32, 64), meta(32, 64))
            model.load_state_dict({k: lumen.from_numpy(v) for k, v in zip(["wq", "wk", "wv"], values)})
            got.append([lumen.to_numpy(v) for v in g(model, *xs)])
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    for merged, alone in zip(*got):
        np.testing.assert_array_equal(merged, alone)


class GatedPair(lumen.nn.Module):
    """SwiGLU's gate and up projections, ``nn.Linear``'s ``[out, in]``
    weights: ``silu(x W1^T) * (x W2^T)``."""

    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        """SwiGLU of `x`."""
        g, u = F.matmul(x, self.w1.t(), "float32", x.dtype), F.matmul(x, self.w2.t(), "float32", x.dtype)
        return g * F.sigmoid(g) * u


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_gated_pairs_weights_are_interleaved(device, triton_on_cuda):
    """Inference: a gated pair's two weights packed into one block
    interleaved (``plan.interleaved``, row j of each its 2j and 2j + 1), the
    pair one kernel (its products' pairs adjacent in a lane), the values as
    unmerged. A training step (its weights assigned, read again by their
    update) interleaves them too: the update reads each weight as a view of
    the block."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        graph = lumen.make_graph(lambda m, x: m(x))(GatedPair(meta(64, 32), meta(64, 32)), meta(16, 32))
        plan = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2], packable=[1, 2])
        assert (plan.packed, plan.interleaved) == ([([1, 2], 0)], [[2]])
        assert len(plan.steps()) == 1, [s["label"] for s in plan.steps()]

    def step(model, x):
        """One SGD step on the sum of the outputs; returns the loss."""
        loss = F.sum(model(x))
        loss.backward()
        for w in (model.w1, model.w2):
            w.copy_(w - 0.1 * w.grad)
        return loss

    if device == "mps":
        graph = lumen.make_graph(step)(GatedPair(meta(64, 32), meta(64, 32)), meta(16, 32))
        plan = lumen.graph_compiler.Plan(
            graph, "mps", parameters=[1, 2], packable=[1, 2], donate_into=[(1, 1), (2, 2)]
        )
        assert plan.interleaved == [[2]], plan.interleaved
    values = {"w1": lumen.from_numpy(rand(64, 32, seed=1)), "w2": lumen.from_numpy(rand(64, 32, seed=2))}
    x = lumen.from_numpy(rand(16, 32, seed=3)).to(device)
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = GatedPair(meta(64, 32), meta(64, 32))
            f = lumen.compile(lambda m, x: m(x), device=device)
            f(model, meta(16, 32))
            model.load_state_dict(values)
            got.append(lumen.to_numpy(f(model, x)))
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    np.testing.assert_allclose(got[0], got[1], rtol=1e-5, atol=1e-6)


class GatedMLP2(lumen.nn.Module):
    """A gated pair, then its down projection (``nn.Linear``'s weights)."""

    w1: lumen.Tensor
    w2: lumen.Tensor
    w3: lumen.Tensor

    def __call__(self, x):
        """Down projection of the SwiGLU of `x`."""
        g, u = F.matmul(x, self.w1.t(), "float32", x.dtype), F.matmul(x, self.w2.t(), "float32", x.dtype)
        return F.matmul(g * F.sigmoid(g) * u, self.w3.t(), "float32", x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_gated_pairs_backward_is_interleaved(device, triton_on_cuda):
    """Gradients of a gated MLP: the pair's block interleaved through its
    backward too (``merge_dots``): its cotangents ``[dg | du]`` interleaved,
    written side by side by the down projection's gradient's epilogue (one
    kernel, ``EXPANDED``), x's gradient one dot of them and the block, the
    weights' one dot, each weight's its pairs. The gradients the unmerged
    program's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    h, d, m = 64, 32, 16

    def step(model, x):
        """The loss and the gradients of `x` and every weight."""
        loss = F.sum(model(x))
        loss.backward()
        return loss, x.grad, model.w1.grad, model.w2.grad, model.w3.grad

    make = lambda: GatedMLP2(meta(h, d), meta(h, d), meta(d, h))
    if device == "mps":
        graph = lumen.make_graph(step)(make(), meta(m, d))
        plan = lumen.graph_compiler.Plan(graph, "mps", parameters=[1, 2, 3], packable=[1, 2, 3])
        assert (plan.packed, plan.interleaved) == ([([1, 2], 0)], [[2]])
        labels = [s["label"] for s in plan.steps()]
        (cotangents,) = [label for label in labels if label.endswith("concatenate")]
        assert cotangents.startswith("dot_general → "), labels
        assert labels.count("2x dot_general") == 2, labels
    values = {"w1": rand(h, d, seed=1) / 4, "w2": rand(h, d, seed=2) / 4, "w3": rand(d, h, seed=3) / 4}
    x = lumen.from_numpy(rand(m, d, seed=4)).to(device)
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = make()
            f = lumen.compile(step, device=device)
            f(model, meta(m, d))
            model.load_state_dict({k: lumen.from_numpy(v) for k, v in values.items()})
            got.append([lumen.to_numpy(t) for t in f(model, x)])
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    for merged, alone in zip(*got):
        np.testing.assert_allclose(merged, alone, rtol=1e-5, atol=1e-5)


class Projections(lumen.nn.Module):
    """Projections of `x` by `ws`, combined in consecutive `groups`; `left` multiplies by the weights
    on the left and `mixed` in bfloat16."""

    ws: list
    groups: list
    mixed: bool = False
    left: bool = False

    def __call__(self, x):
        """One value per group: the negated first projection, gated by each later one in turn."""
        cast = (lambda t: t.bfloat16()) if self.mixed else (lambda t: t)
        if self.left:
            p = [F.matmul(cast(w), cast(x), "float32", "float32") for w in self.ws]
        else:
            p = [F.matmul(cast(x), cast(w.t()), "float32", "float32") for w in self.ws]
        out, k = [], 0
        for g in self.groups:
            q, k = p[k : k + g], k + g
            v = -q[0]
            for r in q[1:]:
                v = v * F.sigmoid(r) + r
            out.append(v)
        return out


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "sizes, groups, mixed, interleaved, steps",
    [
        ([64, 64, 64, 64], [2, 2], False, [[2, 2]], 1),
        ([64, 64, 32, 32], [2, 2], False, [[2, 2]], 1),
        ([64, 64, 32, 32, 16], [2, 2, 1], False, [[2, 2, 1]], 1),
        ([64, 64, 32, 32, 15], [2, 2, 1], False, [[2, 2]], 2),
        ([64, 64, 64, 64], [2, 2], True, [[2, 2]], 1),
        ([64, 64, 64, 64, 64, 64], [2, 4], False, [[4, 2]], 1),
        ([32] * 4, [4], False, [[4]], 1),
        ([32] * 8, [4, 4], False, [[4, 4]], 1),
        ([16] * 8, [8], False, [[8]], 1),
        ([8] * 8 + [16] * 4 + [32] * 2, [8, 4, 2], False, [[8, 4, 2]], 1),
        ([64, 64, 64, 64, 64], [3, 2], False, [[1, 1, 1], [2]], 3),
    ],
    ids=[
        "two pairs",
        "pairs of two sizes",
        "pairs and a member",
        "a member of odd width",
        "mixed precision",
        "a pair and four combined",
        "four",
        "two fours",
        "eight",
        "eight four two",
        "three apart",
    ],
)
@pytest.mark.parametrize("left", [False, True], ids=["x @ w.T", "w @ x"])
def test_dots_of_one_operand_share_one_interleaved_block(
    device, sizes, groups, mixed, interleaved, steps, left, triton_on_cuda
):
    """Dots of one operand, combined in groups of K dividing the 8-column
    tile (``K`` values interleaved: SwiGLU's gate and up for K = 2) and any
    read alone: one block, each group a region of its own interleaved, each
    lone one a region as it is (``plan.interleaved``); one matmul whose
    epilogue computes each group's value from the lane's row. Groups of K
    not dividing the tile, or of a width not, merge apart into a block of
    their own, their epilogues not fused. Weights on the left (``w @ x``)
    interleave along the output's rows instead. The values the unmerged
    program's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    d, m = 32, 16
    n = len(sizes)
    make = lambda: Projections([meta(h, d) for h in sizes], groups, mixed, left)
    shape = (d, m) if left else (m, d)
    if device == "mps":
        graph = lumen.make_graph(lambda model, x: model(x))(make(), meta(*shape))
        params = list(range(1, n + 1))
        plan = lumen.graph_compiler.Plan(graph, "mps", parameters=params, packable=params)
        assert plan.interleaved == interleaved, plan.packed
        labels = [s["label"] for s in plan.steps() if not re.match(r"(\d+x )?cast", s["label"])]
        assert len(labels) == steps, labels
        for regions in interleaved:
            assert any(label.startswith(f"{sum(regions)}x dot_general") for label in labels), labels
    values = {f"ws.{k}": lumen.from_numpy(rand(h, d, seed=k + 1) / 4) for k, h in enumerate(sizes)}
    x = lumen.from_numpy(rand(*shape, seed=9)).to(device)
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = make()
            f = lumen.compile(lambda model, x: model(x), device=device)
            f(model, meta(*shape))
            model.load_state_dict(values)
            got.append([lumen.to_numpy(t) for t in f(model, x)])
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    for merged, alone in zip(*got):
        np.testing.assert_allclose(merged, alone, rtol=1e-5, atol=1e-6)


class SharedOperand(lumen.nn.Module):
    """Projections of `x` sharing it as an operand, combined in consecutive `groups`."""

    ws: list
    groups: list
    left: bool = False

    def __call__(self, x):
        """One value per group: silu of its first projection plus the others."""
        if self.left:
            p = [F.matmul(w, x, "float32", "float32") for w in self.ws]
        else:
            p = [F.matmul(x, w.t(), "float32", "float32") for w in self.ws]
        out, k = [], 0
        for g in self.groups:
            q, k = p[k : k + g], k + g
            v = q[0] * F.sigmoid(q[0])
            for r in q[1:]:
                v = v + r
            out.append(v)
        return out


def _gated_inference(model, x):
    """Forward only."""
    return model(x)


def _gated_gradients(model, x):
    """The summed squared outputs, `x`'s gradient and every weight's."""
    loss = sum(F.sum(o * o) for o in model(x))
    loss.backward()
    return [loss, x.grad] + [w.grad for w in model.ws]


def _gated_training(model, x):
    """One SGD step on the summed squared outputs; returns the loss."""
    loss = sum(F.sum(o * o) for o in model(x))
    loss.backward()
    for w in model.ws:
        w.copy_(w - 0.01 * w.grad)
    return loss


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "groups, h, interleaved",
    [
        ([1, 1, 1], 64, [[1, 1, 1]]),
        ([2], 64, [[2]]),
        ([4], 64, [[4]]),
        ([8], 64, [[8]]),
        ([2, 2], 64, [[2, 2]]),
        ([4, 4], 32, [[4, 4]]),
        ([8, 4, 2, 1], 32, [[8, 4, 2, 1]]),
        ([3], 64, [[1, 1, 1]]),
        ([2, 1], 64, [[2, 1]]),
        ([2], 10, [[1, 1]]),
    ],
    ids=[
        "ones",
        "pair",
        "four",
        "eight",
        "two pairs",
        "two fours",
        "eight four two one",
        "three",
        "pair and one",
        "pair apart",
    ],
)
@pytest.mark.parametrize(
    "step", [_gated_inference, _gated_gradients, _gated_training], ids=["inference", "gradients", "training"]
)
def test_weights_shared_on_either_side_merge_alike(device, groups, h, interleaved, step, triton_on_cuda):
    """Dots sharing their operand merge the same whether the weights are
    the right operand (``x @ w.T``) or the left (``w @ x``): the same
    block layout and the same kernels, forward and backward, the left
    form's blocks interleaved along the output's rows, a pair too narrow
    to interleave read as row halves. Every weight is packed. The values,
    gradients and trained weights the unmerged program's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    d, m = 32, 24
    n = sum(groups)
    params = list(range(1, n + 1))
    plans = []
    for left in (False, True):
        shape = (d, m) if left else (m, d)
        make = lambda: SharedOperand([meta(h, d) for _ in range(n)], groups, left)
        graph = lumen.make_graph(step)(make(), meta(*shape))
        plan = lumen.graph_compiler.Plan(graph, device, parameters=params, packable=params)
        assert sorted(k for members, _ in plan.packed for k in members) == params, plan.packed
        if device == "mps":
            assert plan.interleaved == interleaved
            plans.append(sorted(s["label"] for s in plan.steps()))
        values = {f"ws.{k}": lumen.from_numpy(rand(h, d, seed=k + 1) / 4) for k in range(n)}
        x = lumen.from_numpy(rand(*shape, seed=9)).to(device)
        got = []
        try:
            for merge in (True, False):
                getattr(lumen.config.compiler, device).merge_dots = merge
                model = make()
                f = lumen.compile(step, device=device)
                f(model, meta(*shape))
                model.load_state_dict(values)
                for _ in range(2):
                    out = f(model, x)
                out = out if isinstance(out, (list, tuple)) else [out]
                got.append([lumen.to_numpy(t) for t in out] + [lumen.to_numpy(w._placed(device)) for w in model.ws])
        finally:
            getattr(lumen.config.compiler, device).merge_dots = True
        for merged, alone in zip(*got):
            assert np.isfinite(alone).all()
            np.testing.assert_allclose(merged, alone, rtol=1e-4, atol=1e-5 * np.abs(alone).max())
    if device == "mps":
        assert plans[0] == plans[1]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "make, n",
    [
        *[
            (lambda ws, left, g=g: SharedOperand(ws, g, left), sum(g))
            for g in ([2], [4], [8], [2, 2], [2, 1], [8, 4, 2, 1])
        ],
        (lambda ws, left: Returned(ws, True, left), 4),
        (lambda ws, left: Returned(ws, False, left), 2),
    ],
    ids=["pair", "four", "eight", "two pairs", "pair and one", "eight four two one", "y = Ax + Bx, Cx, Dx", "Cx, Dx"],
)
@pytest.mark.parametrize("left", [False, True], ids=["x @ w.T", "w @ x"])
def test_merged_regions_fuse(device, make, n, left, triton_on_cuda):
    """A merged matmul's regions (interleaved groups, plain slices, results
    returned as they are) run in its GEMM's epilogue, the weights
    on either side: one step, each region writing its own output; the
    values the unmerged program's."""
    d, m, h = 32, 24, 64
    params = list(range(1, n + 1))
    shape = (d, m) if left else (m, d)
    try:
        x = lumen.from_numpy(rand(*shape, seed=9)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(lambda model, x: model(x))(make([meta(h, d) for _ in range(n)], left), meta(*shape))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=params, packable=params)
    assert len(plan.steps()) == 1, [s["label"] for s in plan.steps()]
    values = {f"ws.{k}": lumen.from_numpy(rand(h, d, seed=k + 1) / 4) for k in range(n)}
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = make([meta(h, d) for _ in range(n)], left)
            f = lumen.compile(lambda model, x: model(x), device=device)
            f(model, meta(*shape))
            model.load_state_dict(values)
            out = f(model, x)
            assert all(t.is_contiguous() for t in out)
            got.append([lumen.to_numpy(t.to("cpu")) for t in out])
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    for merged, alone in zip(*got):
        np.testing.assert_allclose(merged, alone, rtol=1e-5, atol=1e-5 * np.abs(alone).max())


class Returned(lumen.nn.Module):
    """Projections of `x`, returned each on its own, or with the first two summed."""

    ws: list
    summed: bool
    left: bool = False

    def __call__(self, x):
        """The projections, the first two added when `summed`."""
        if self.left:
            p = [F.matmul(w, x, "float32", "float32") for w in self.ws]
        else:
            p = [F.matmul(x, w.t(), "float32", "float32") for w in self.ws]
        return (p[0] + p[1], *p[2:]) if self.summed else tuple(p)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("left", [False, True], ids=["x @ w.T", "w @ x"])
@pytest.mark.parametrize(
    "n, summed, interleaved",
    [(4, True, [[2, 1, 1]]), (2, False, [[1, 1]])],
    ids=["y = Ax + Bx, Cx, Dx", "Cx, Dx"],
)
def test_returned_dots_merge(device, n, summed, interleaved, left, triton_on_cuda):
    """Dots whose results are returned merge too: one matmul whose
    epilogue writes each returned result into its own contiguous output
    (``y = Ax + Bx`` from the interleaved pair, ``Cx`` and ``Dx`` from
    regions of their own), no copy after it. The values the unmerged
    program's."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    h, d, m = 64, 32, 24
    shape = (d, m) if left else (m, d)
    params = list(range(1, n + 1))
    make = lambda: Returned([meta(h, d) for _ in range(n)], summed, left)
    if device == "mps":
        graph = lumen.make_graph(lambda model, x: model(x))(make(), meta(*shape))
        plan = lumen.graph_compiler.Plan(graph, "mps", parameters=params, packable=params)
        assert plan.interleaved == interleaved
        labels = [s["label"] for s in plan.steps()]
        assert len(labels) == 1 and labels[0].startswith(f"{n}x dot_general"), labels
    values = {f"ws.{k}": lumen.from_numpy(rand(h, d, seed=k + 1)) for k in range(n)}
    x = lumen.from_numpy(rand(*shape, seed=9)).to(device)
    got = []
    try:
        for merge in (True, False):
            getattr(lumen.config.compiler, device).merge_dots = merge
            model = make()
            f = lumen.compile(lambda model, x: model(x), device=device)
            f(model, meta(*shape))
            model.load_state_dict(values)
            out = f(model, x)
            assert all(t.is_contiguous() for t in out)
            got.append([lumen.to_numpy(t) for t in out])
    finally:
        getattr(lumen.config.compiler, device).merge_dots = True
    for merged, alone in zip(*got):
        np.testing.assert_allclose(merged, alone, rtol=1e-5, atol=1e-5 * np.abs(alone).max())


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("module", [QKV, MixedQKV], ids=["float32", "mixed precision"])
def test_trained_projections_merge_into_one_matmul(device, module, triton_on_cuda):
    """Weights a training step assigns (its optimizer's update, written over
    them in place) merge too, as PyTorch's fused QKV weight: ``[out, in]``
    weights side by side along their first dimension, each a contiguous
    part of the block, updated there in place once no matmul reads the
    block (the planner's block-aware donation). The weights after a few
    steps are the unmerged program's. In mixed precision too: the weights'
    bfloat16 casts one cast of their block."""
    from lumen.profiler import ProfilerActivity, profile

    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def step(model, x, c):
        """One SGD step on a loss combining q, k and v; returns the loss."""
        q, k, v = model(x)
        loss = F.sum((q * k + v) * c)
        loss.backward()
        for w in (model.wq, model.wk, model.wv):
            w.copy_(w - 0.1 * w.grad)
        return loss

    x, c = lumen.from_numpy(rand(16, 32)), lumen.from_numpy(rand(16, 32, seed=9))
    values = {n: lumen.from_numpy(rand(32, 32, seed=i + 1) / 4) for i, n in enumerate(("wq", "wk", "wv"))}
    trained = {}
    for merge in (True, False):
        getattr(lumen.config.compiler, device).merge_dots = merge
        try:
            model = module(meta(32, 32), meta(32, 32), meta(32, 32))
            f = lumen.compile(step, device=device)
            f(model, meta(16, 32), meta(16, 32))
            model.load_state_dict(values)
            if device == "mps":
                with profile(activities=[ProfilerActivity.MPS]) as prof:
                    for _ in range(3):
                        f(model, x, c)
                    lumen.mps.synchronize()
            else:
                for _ in range(3):
                    f(model, x, c)
        finally:
            getattr(lumen.config.compiler, device).merge_dots = True
        if device == "mps":
            kernels = [e["name"] for e in prof.events() if e["kind"] == "gpu"]
            assert ("3x dot_general" in kernels) == merge, kernels
        trained[merge] = [lumen.to_numpy(w._placed(device)) for w in (model.wq, model.wk, model.wv)]
    for merged, unmerged in zip(trained[True], trained[False]):
        np.testing.assert_allclose(merged, unmerged, rtol=1e-5, atol=1e-6)


class Linear(lumen.nn.Module):
    """Matmul by `w`."""

    w: lumen.Tensor

    def __call__(self, x):
        """`x @ w`."""
        return F.matmul(x, self.w, "float32", x.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_packed_weights_are_read_in_place_elsewhere(device, triton_on_cuda):
    """A weight another function packed into a block (a strided view of it)
    is read in place by the matmuls of a function using it alone, at its
    strides: no copy per call. Large enough to be tuned (a few rows, the
    skinny kernels), the tuned kernel, which assumes whole operands, is
    not used for the strided one."""
    from lumen.profiler import ProfilerActivity, profile

    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    x, w1v, w3v = rand(16, 1024), rand(1024, 1024, seed=1), rand(1024, 1024, seed=2)
    gated = Gated(meta(1024, 1024), meta(1024, 1024))
    run = lumen.compile(lambda model, x: model(x), device=device)
    run(gated, meta(16, 1024))
    gated.load_state_dict({"w1": lumen.from_numpy(w1v), "w3": lumen.from_numpy(w3v)})
    alone = Linear(gated.w1)
    run(alone, lumen.from_numpy(x))
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = run(alone, lumen.from_numpy(x))
    np.testing.assert_allclose(lumen.to_numpy(out), x @ w1v, rtol=1e-4, atol=1e-4)
    assert "lumen::to_vec" not in {e["name"] for e in prof.events()}
