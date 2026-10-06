"""Attention: F.flash_attention ([B, S, N, H], grouped-query,
causal), and the MPS compiler running attention, however written, as one
flash-attention kernel (lumen/compiler/mps/attention.rs, attention.metal)."""

import warnings

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

MPS = pytest.param("mps", marks=pytest.mark.mps)


def reference(q, k, v, scale=None, causal=False):
    """Attention in float64: q [B, Sq, N, H], k and v [B, Sk, Nkv, *]; key j
    seen by query i iff j <= i + Sk - Sq when causal."""
    q, k, v = (a.astype(np.float64) for a in (q, k, v))
    b, sq, n, h = q.shape
    sk, nkv = k.shape[1], k.shape[2]
    k, v = np.repeat(k, n // nkv, axis=2), np.repeat(v, n // nkv, axis=2)
    s = np.einsum("bqnh,bknh->bnqk", q, k) * (1 / np.sqrt(h) if scale is None else scale)
    if causal:
        i, j = np.arange(sq)[:, None], np.arange(sk)[None, :]
        s = np.where(j <= i + sk - sq, s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bnqk,bknh->bqnh", p, v)


def tensors(device, dtype, *shapes, seed=0):
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    try:
        ts = [lumen.from_numpy(a).to(device).to(dtype=dtype) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    # The values as rounded to `dtype`, for the reference.
    return ts, [lumen.to_numpy(t.to(dtype="float32")) for t in ts]


def steps(f, *args):
    return [s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(*args), "mps").steps()]


TOL = {"float32": 2e-6, "float16": 2e-3, "bfloat16": 2e-2}

# (B, Sq, Sk, N, Nkv, H, Hv), causal: plain, ragged sizes, grouped-query,
# causal, decoding (one and few queries), head sizes that differ, and a
# key length past one block.
CASES = [
    ((2, 100, 100, 4, 4, 64, 64), False),
    ((1, 130, 77, 2, 2, 32, 32), False),
    ((2, 100, 100, 4, 2, 64, 64), True),
    ((1, 1, 300, 8, 2, 128, 128), True),
    ((2, 5, 40, 2, 1, 16, 24), False),
    ((1, 70, 200, 2, 2, 128, 128), True),
]


# The CPU (the reference executor, as traced) on smaller ones.
CPU_CASES = [
    ((2, 9, 9, 4, 2, 8, 8), True),
    ((1, 6, 11, 2, 2, 16, 8), False),
    ((1, 1, 13, 4, 1, 8, 8), True),
]


def _ids(cases):
    return [f"{c[0]}{'-causal' if c[1] else ''}" for c in cases]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "device, case",
    [("cpu", c) for c in CPU_CASES] + [pytest.param("mps", c, marks=pytest.mark.mps) for c in CASES],
    ids=[f"cpu-{i}" for i in _ids(CPU_CASES)] + [f"mps-{i}" for i in _ids(CASES)],
)
def test_flash_attention(device, case, dtype):
    """F.flash_attention agrees with attention in float64, on
    the CPU (as traced) and on MPS (one flash-attention kernel)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    (q, k, v), arrays = tensors(device, dtype, (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv))

    def f(q, k, v):
        return F.flash_attention(q, k, v, is_causal=causal)

    out = lumen.compile(f)(q, k, v)
    assert out.dtype == dtype and list(out.shape) == [b, sq, n, hv]
    np.testing.assert_allclose(
        lumen.to_numpy(out.to(dtype="float32")), reference(*arrays, causal=causal), rtol=TOL[dtype], atol=TOL[dtype]
    )
    if device == "mps":
        assert steps(f, q, k, v) == ["flash_attention"]


@pytest.mark.mps
def test_attention_as_traced_without_flash_attention():
    """With ``flash_attention`` off, attention runs as traced (scores in
    memory, several kernels); with it on (the default), one kernel, agreeing
    to rounding."""
    (q, k, v), _ = tensors("mps", "float16", (1, 64, 4, 32), (1, 64, 2, 32), (1, 64, 2, 32))

    def f(q, k, v):
        return F.flash_attention(q, k, v, is_causal=True)

    assert lumen.config.compiler.flash_attention is True
    flash = lumen.to_numpy(lumen.compile(f)(q, k, v).to(dtype="float32"))
    try:
        lumen.config.compiler.flash_attention = False
        assert len(steps(f, q, k, v)) > 1 and "flash_attention" not in steps(f, q, k, v)
        traced = lumen.to_numpy(lumen.compile(f)(q, k, v).to(dtype="float32"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_allclose(flash, traced, rtol=4e-3, atol=4e-3)


def _written(q, k, v):
    """Attention as a model writes it: [B, N, S, H], grouped-query heads by
    expanding K, a scale, the softmax in float32 and cast back."""
    b, n, s, h = q.shape
    nkv = k.shape[1]
    k = k.unsqueeze(2).expand(b, nkv, n // nkv, s, h).reshape(b, n, s, h)
    scores = (q @ k.transpose(-1, -2)) * (1.0 / h**0.5)
    return F.softmax(scores.float(), -1).to(dtype=q.dtype) @ v


@pytest.mark.mps
def test_attention_written_out_is_one_kernel():
    """Attention written out (``_written``) is matched too: one kernel,
    agreeing with the CPU; at a model's size too."""
    (q, k, v), arrays = tensors("mps", "bfloat16", (2, 4, 24, 16), (2, 2, 24, 16), (2, 4, 24, 16))
    assert steps(_written, q, k, v) == ["flash_attention"]
    got = lumen.to_numpy(lumen.compile(_written)(q, k, v).to(dtype="float32"))
    cpu = [lumen.from_numpy(a).to(dtype="bfloat16") for a in arrays]
    want = lumen.to_numpy(lumen.compile(_written, device="cpu")(*cpu).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)
    big = [
        lumen.empty(s, dtype="bfloat16", device="meta")
        for s in ([2, 32, 1024, 128], [2, 8, 1024, 128], [2, 32, 1024, 128])
    ]
    assert steps(_written, *big) == ["flash_attention"]


@pytest.mark.mps
def test_attention_the_kernels_cannot_take_runs_as_traced():
    """A head dimension past the kernels' (256) or not a multiple of 8 runs
    as traced, still correct."""
    for h in (512, 20):
        (q, k, v), arrays = tensors("mps", "float32", (1, 16, 2, h), (1, 16, 2, h), (1, 16, 2, h))

        def f(q, k, v):
            return F.flash_attention(q, k, v)

        assert "flash_attention" not in steps(f, q, k, v)
        out = lumen.to_numpy(lumen.compile(f)(q, k, v))
        np.testing.assert_allclose(out, reference(*arrays), rtol=1e-5, atol=1e-5)


def test_flash_attention_checks_its_operands():
    def call(q, k, v):
        return lumen.make_graph(F.flash_attention)(q, k, v)

    m = lambda *s, dtype="float32": lumen.empty(list(s), dtype=dtype, device="meta")  # noqa: E731
    with pytest.raises(ValueError, match="multiple of Nkv"):
        call(m(1, 4, 3, 8), m(1, 4, 2, 8), m(1, 4, 2, 8))
    with pytest.raises(ValueError, match=r"\[B, S, N, H\]"):
        call(m(4, 3, 8), m(4, 3, 8), m(4, 3, 8))
    with pytest.raises(TypeError, match="dtypes"):
        call(m(1, 4, 2, 8), m(1, 4, 2, 8, dtype="float16"), m(1, 4, 2, 8))
    with pytest.raises(TypeError, match="floating-point"):
        call(m(1, 4, 2, 8, dtype="int32"), m(1, 4, 2, 8, dtype="int32"), m(1, 4, 2, 8, dtype="int32"))


@pytest.mark.mps
@pytest.mark.parametrize(
    "f, sq",
    [
        (lambda q, k, v, x, b: x + F.flash_attention(q, k, v).reshape(-1, 128), 96),
        (
            lambda q, k, v, x, b: (F.flash_attention(q, k, v, is_causal=True).reshape(-1, 128) * 2.0 + b).to(
                dtype="float16"
            ),
            96,
        ),
        (lambda q, k, v, x, b: x + F.flash_attention(q, k, v, is_causal=True).reshape(-1, 128), 1),
    ],
    ids=["residual", "scale bias cast", "decode residual"],
)
def test_attention_epilogues_fuse(f, sq):
    """The elementwise primitives after an attention, through reshapes (a
    residual, a bias, a scale, a cast), run in its kernel as it writes each
    output (``contraction_epilogues``): one step, labeled with them,
    agreeing with the CPU and with the unfused plan."""
    shapes = [(1, sq, 4, 32), (1, 96, 4, 32), (1, 96, 4, 32), (sq, 128), (128,)]
    ts, arrays = tensors("mps", "float32", *shapes)
    # Profiled as the attention and its epilogue: flash_attention → reshape → ...
    (label,) = steps(f, *ts)
    assert label.startswith("flash_attention → reshape → "), label
    got = lumen.to_numpy(lumen.compile(f)(*ts).to(dtype="float32"))
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(*(lumen.from_numpy(a) for a in arrays)).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    lumen.config.compiler.contraction_epilogues = False
    try:
        assert len(steps(f, *ts)) == 2
        unfused = lumen.to_numpy(lumen.compile(f)(*ts).to(dtype="float32"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_allclose(got, unfused, rtol=1e-6, atol=1e-6)


def _sources(f, *args):
    plan = lumen.graph.Plan(lumen.make_graph(f)(*args), "mps")
    return {s["label"]: (s.get("fusion") or {}).get("source", "") for s in plan.steps()}


@pytest.mark.mps
def test_attention_computes_its_scores_as_traced():
    """The kernel rounds the scores as traced: ``_written``'s ``q @ k^T`` to
    bf16, its scale in bf16, then cast to float32 for the softmax;
    F.flash_attention's in float32."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([1, 4, 64, 32], [1, 2, 64, 32], [1, 4, 64, 32])]
    source = _sources(_written, *m)["flash_attention"]
    assert "convert_value<float>(Mul::apply(convert_value<bfloat>(s), as_type<bfloat>" in source, source
    m = [lumen.empty([1, 64, 4, 32], dtype="bfloat16", device="meta")] * 3
    source = _sources(F.flash_attention, *m)["flash_attention"]
    assert "(convert_value<float>(s), as_type<float>" in source, source


@pytest.mark.mps
@pytest.mark.parametrize(
    "f",
    [
        # The softmax in f16.
        lambda q, k, v: F.softmax(q.half() @ k.half().transpose(-1, -2), -1).to(dtype=q.dtype) @ v,
        # The scores accumulated in bf16.
        lambda q, k, v: F.softmax(
            prims.dot_general(q, k, (((3,), (3,)), ((0, 1), (0, 1))), "bfloat16", "bfloat16").float(), -1
        ).to(dtype=q.dtype)
        @ v,
    ],
    ids=["f16 softmax", "bf16 accumulation"],
)
def test_attention_the_kernel_cannot_compute_as_traced_runs_as_traced(f):
    """Attention computing in a precision other than the kernel's (float32
    softmax and accumulation) is not matched: it runs as traced."""
    m = [lumen.empty([1, 4, 64, 32], dtype="bfloat16", device="meta")] * 3
    assert "flash_attention" not in _sources(f, *m)


# Smaller (the CPU's gradient is the reference): grouped-query and causal,
# a query past the key block, few queries (decoding), head sizes that
# differ, the backward kernels' largest head.
BACKWARD_CASES = [
    ((1, 40, 40, 4, 2, 32, 32), True),
    ((1, 70, 20, 2, 2, 16, 16), False),
    ((1, 3, 50, 2, 1, 16, 24), True),
    ((1, 9, 24, 2, 2, 128, 128), True),
]


def backward_kernels(h, hv, dtype, deterministic=False):
    """An attention backward's kernels: one adding dQ atomically, if its
    blocks fit in threadgroup memory (codegen.rs's
    ``backward_block_with_dq``) and not ``deterministic``; else two."""
    t = {"float32": 4, "bfloat16": 2}[dtype]
    fits = any(n * (h + hv) * t + 8 * n + 64 * h * t + 64 * n * t <= 28 << 10 for n in (32, 16))
    if fits and not deterministic:
        return ["flash_attention_backward(dk, dv, dq)"]
    return ["flash_attention_backward(dk, dv)", "flash_attention_backward(dq)"]


@pytest.mark.mps
@pytest.mark.parametrize("deterministic", [False, True], ids=["", "deterministic"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("device, case", [("mps", c) for c in BACKWARD_CASES], ids=_ids(BACKWARD_CASES))
def test_attention_backward_is_flash_attention(device, case, dtype, deterministic):
    """The gradient of F.flash_attention on MPS: the forward one
    flash-attention kernel (writing each row's log-sum-exp too), the
    backward one adding dQ atomically by its dK and dV kernel (where its
    blocks fit), or two (dK and dV, then dQ; ``deterministic``: the same
    bits each run), recomputing the probabilities; agreeing with the CPU's
    (the same program, as traced)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    shapes = (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv), (b, sq, n, hv)
    (q, k, v, w), arrays = tensors(device, dtype, *shapes)

    def loss(q, k, v, w):
        return F.sum((F.flash_attention(q, k, v, is_causal=causal) * w).float())

    grad = lumen.grad(loss, (0, 1, 2))
    lumen.config.compiler.deterministic = deterministic
    try:
        labels = steps(grad, q, k, v, w)
        for kernel in ["flash_attention", *backward_kernels(h, hv, dtype, deterministic)]:
            assert kernel in labels, labels
        compiled = lumen.compile(grad)
        got = compiled(q, k, v, w)
        if deterministic:
            for g, again in zip(got, compiled(q, k, v, w)):
                g, again = (lumen.to_numpy(t.to(dtype="float32")) for t in (g, again))
                np.testing.assert_array_equal(g, again)
    finally:
        lumen.config.compiler.reset()
    cpu = [lumen.from_numpy(a).to(dtype=dtype) for a in arrays]
    want = lumen.compile(grad, device="cpu")(*cpu)
    tol = {"float32": 1e-5, "bfloat16": 2e-2}[dtype]
    for g, c in zip(got, want):
        g, c = (lumen.to_numpy(t.to(dtype="float32")) for t in (g, c))
        np.testing.assert_allclose(g, c, rtol=tol, atol=tol * np.abs(c).max())


@pytest.mark.mps
@pytest.mark.parametrize("deterministic", [False, True], ids=["", "deterministic"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("attention", [F.flash_attention, F.naive_attention], ids=["flash", "naive"])
@pytest.mark.parametrize("case", BACKWARD_CASES, ids=_ids(BACKWARD_CASES))
def test_attention_dropout_is_in_its_kernels(case, attention, dtype, deterministic):
    """``dropout_p`` (F.flash_attention's, or F.dropout of the probabilities
    written out, as F.naive_attention does) runs in the attention's own
    kernels on MPS, forward and backward: each draws the mask's random bits
    where it reads a probability, none stored (no other kernel); the loss
    and gradient the CPU's (the same bits, the program as traced)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    shapes = (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv), (b, sq, n, hv)
    (q, k, v, w), arrays = tensors("mps", dtype, *shapes)

    def loss(q, k, v, w):
        return F.sum((attention(q, k, v, is_causal=causal, dropout_p=0.25) * w).float())

    step = lumen.value_and_grad(loss, (0, 1, 2))
    lumen.config.compiler.deterministic = deterministic
    try:
        labels = steps(step, q, k, v, w)
        # The forward's label has its epilogue's primitives after it.
        kernels = [label.split(" → ")[0] for label in labels]
        for kernel in ["flash_attention", *backward_kernels(h, hv, dtype, deterministic)]:
            assert kernel in kernels, labels
        assert not [label for label in labels if "random_bits" in label], labels
        lumen.manual_seed(3)
        got = lumen.compile(step)(q, k, v, w)
    finally:
        lumen.config.compiler.reset()
    lumen.manual_seed(3)
    cpu = [lumen.from_numpy(a).to(dtype=dtype) for a in arrays]
    want = lumen.compile(step, device="cpu")(*cpu)
    tol = {"float32": 1e-5, "bfloat16": 2e-2}[dtype]
    for g, c in zip([got[0], *got[1]], [want[0], *want[1]]):
        g, c = (lumen.to_numpy(t.to(dtype="float32")) for t in (g, c))
        np.testing.assert_allclose(g, c, rtol=tol, atol=tol * np.abs(c).max())


def test_attention_dropout_drops_the_probabilities():
    """F.flash_attention's ``dropout_p``: F.dropout of the probabilities (as
    F.naive_attention's, the same bits from the same seed); 0 is none; it
    must be below 1."""
    (q, k, v), _ = tensors("cpu", "float32", (1, 6, 2, 8), (1, 6, 2, 8), (1, 6, 2, 8))
    run = {}
    for name, fn, p in [
        ("flash", F.flash_attention, 0.5),
        ("naive", F.naive_attention, 0.5),
        ("none", F.flash_attention, 0.0),
    ]:
        lumen.manual_seed(0)
        run[name] = lumen.to_numpy(lumen.compile(lambda q, k, v: fn(q, k, v, dropout_p=p))(q, k, v))
    np.testing.assert_allclose(run["flash"], run["naive"], rtol=1e-6, atol=1e-6)
    assert not np.allclose(run["flash"], run["none"])
    with pytest.raises(ValueError, match="dropout_p"):
        lumen.compile(lambda q, k, v: F.flash_attention(q, k, v, dropout_p=1.0))(q, k, v)


@pytest.mark.mps
def test_written_attention_trains_with_flash_attention():
    """Attention written out (``_written``) is flash attention in training
    too: its gradient's plan has the forward and both backward kernels."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([2, 4, 64, 32], [2, 2, 64, 32], [2, 4, 64, 32])]

    def loss(q, k, v):
        return F.sum(_written(q, k, v).float())

    labels = steps(lumen.grad(loss, (0, 1, 2)), *m)
    for kernel in ["flash_attention", *backward_kernels(32, 32, "bfloat16")]:
        assert kernel in labels, labels


@pytest.mark.mps
@pytest.mark.parametrize("scale", [1.0, 0.125], ids=["unscaled", "scaled"])
def test_written_attention_backward_is_flash_attention(scale):
    """Attention written out trains with flash attention's backward kernels
    whether its scores are scaled or not (``* 1`` simplified away, dS has no
    scale to multiply by: 1); the gradients the CPU's."""
    (q, k, v), arrays = tensors("mps", "float32", (2, 40, 32), (2, 40, 32), (2, 40, 32))

    def loss(q, k, v):
        return F.sum(F.tanh(F.softmax((q @ k.transpose(-1, -2)) * scale, -1) @ v))

    grad = lumen.grad(loss, (0, 1, 2))
    labels = steps(grad, q, k, v)
    # The forward's label has its epilogue's primitives after it.
    kernels = [label.split(" → ")[0] for label in labels]
    for kernel in ["flash_attention", *backward_kernels(32, 32, "float32")]:
        assert kernel in kernels, labels
    got = lumen.compile(grad)(q, k, v)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("causal", [False, True], ids=["", "causal"])
def test_naive_attention_is_flash_attention(causal):
    """F.naive_attention (attention as main first wrote it) computes as
    F.flash_attention does, its gradient too."""
    (q, k, v), _ = tensors("cpu", "float32", (2, 7, 4, 8), (2, 9, 2, 8), (2, 9, 2, 8))
    for fn in (lambda f: f, lambda f: lumen.grad(lambda q, k, v: F.sum(F.tanh(f(q, k, v))), (0, 1, 2))):
        naive = lumen.compile(fn(lambda q, k, v: F.naive_attention(q, k, v, is_causal=causal)))(q, k, v)
        sdpa = lumen.compile(fn(lambda q, k, v: F.flash_attention(q, k, v, is_causal=causal)))(q, k, v)
        for a, b in zip(naive if isinstance(naive, tuple) else [naive], sdpa if isinstance(sdpa, tuple) else [sdpa]):
            np.testing.assert_allclose(lumen.to_numpy(a), lumen.to_numpy(b), rtol=1e-6, atol=1e-6)


@pytest.mark.mps
@pytest.mark.parametrize("sq", [40, 1], ids=["tiled", "decode"])
def test_attention_epilogue_fuses_in_training(sq):
    """In a training step the attention's value is read by its epilogue (a
    residual) and by the backward: the epilogue still runs in its kernel,
    which writes the attention's value too; the step agrees exactly with
    the unfused one (``deterministic``: dQ in a fixed order)."""
    lumen.config.compiler.deterministic = True
    shapes = [(1, sq, 4, 32), (1, 40, 2, 32), (1, 40, 2, 32), (sq, 128)]
    ts, _ = tensors("mps", "float32", *shapes)

    def block(q, k, v, x):
        return F.sum(F.tanh(x + F.flash_attention(q, k, v, is_causal=True).reshape(sq, 128)))

    step = lumen.value_and_grad(block, (0, 1, 2, 3))
    labels = steps(step, *ts)
    assert "flash_attention → reshape → add → tanh" in labels, labels
    assert "flash_attention_backward(dk, dv)" in labels and "flash_attention_backward(dq)" in labels
    loss, grads = lumen.compile(step)(*ts)
    got = [lumen.to_numpy(t) for t in (loss, *grads)]
    lumen.config.compiler.contraction_epilogues = False
    try:
        loss, grads = lumen.compile(step)(*ts)
    finally:
        lumen.config.compiler.reset()
    for g, w in zip(got, (loss, *grads)):
        np.testing.assert_array_equal(g, lumen.to_numpy(w))


@pytest.mark.mps
def test_log_sum_exp_is_written_only_for_training():
    """The forward kernel writes each row's log-sum-exp only when a backward
    reads it: without one, it is pruned before compiling."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([1, 64, 4, 32], [1, 64, 2, 32], [1, 64, 2, 32])]

    def forward(q, k, v):
        return F.flash_attention(q, k, v, is_causal=True)

    def source(f):
        plan = lumen.graph.Plan(lumen.make_graph(f)(*m), "mps")
        (step,) = [s for s in plan.steps() if s["label"] == "flash_attention"]
        return step["fusion"]["source"]

    assert "float *lse" not in source(forward)
    assert "float *lse" in source(lumen.grad(lambda q, k, v: F.sum(forward(q, k, v).float()), (0, 1, 2)))


def test_flash_attention_training_has_no_precision_warning():
    """In bf16, flash attention's backward rounds nothing to bf16 only to
    widen it (D = rowsum(dO * O) a dot accumulating in float32): no
    precision warning."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([1, 16, 4, 32], [1, 16, 2, 32], [1, 16, 2, 32])]

    def loss(q, k, v):
        return F.sum(F.flash_attention(q, k, v, is_causal=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lumen.make_graph(lumen.grad(loss, (0, 1, 2)))(*m)
