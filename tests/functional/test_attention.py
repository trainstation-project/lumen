"""Tests of flash attention, decode attention and their backward passes against a float64 reference,
and of which written-out attentions the compilers match into the fused kernels."""

import warnings

import numpy as np
import pytest
from helpers import CUDA, MPS

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile


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
    """Random float32 arrays of `shapes` as tensors of `dtype` on `device`, with their values as float32 arrays.

    Skips the test when the device is unavailable."""
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    try:
        ts = [lumen.from_numpy(a).to(device).to(dtype=dtype) for a in arrays]
    except RuntimeError as e:
        pytest.skip(str(e))
    return ts, [lumen.to_numpy(t.to(dtype="float32")) for t in ts]


def steps(f, *args, device="mps"):
    """Step labels of `f` traced on `args` and planned for `device`."""
    return [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]


TOL = {"float32": 2e-6, "float16": 2e-3, "bfloat16": 2e-2}

CASES = [
    ((2, 100, 100, 4, 4, 64, 64), False),
    ((1, 130, 77, 2, 2, 32, 32), False),
    ((2, 100, 100, 4, 2, 64, 64), True),
    ((1, 1, 300, 8, 2, 128, 128), True),
    ((2, 5, 40, 2, 1, 16, 24), False),
    ((1, 70, 200, 2, 2, 128, 128), True),
]


CPU_CASES = [
    ((2, 9, 9, 4, 2, 8, 8), True),
    ((1, 6, 11, 2, 2, 16, 8), False),
    ((1, 1, 13, 4, 1, 8, 8), True),
]


def _ids(cases):
    """Test ids of attention `cases`: the shape tuple, marked when causal."""
    return [f"{c[0]}{'-causal' if c[1] else ''}" for c in cases]


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize(
    "device, case",
    [("cpu", c) for c in CPU_CASES]
    + [pytest.param("mps", c, marks=pytest.mark.mps) for c in CASES]
    + [pytest.param("cuda", c, marks=pytest.mark.cuda) for c in CASES]
    + [pytest.param("tpu", c, marks=pytest.mark.tpu) for c in CASES],
    ids=[f"cpu-{i}" for i in _ids(CPU_CASES)]
    + [f"mps-{i}" for i in _ids(CASES)]
    + [f"cuda-{i}" for i in _ids(CASES)]
    + [f"tpu-{i}" for i in _ids(CASES)],
)
def test_flash_attention(device, case, dtype, triton_on_cuda):
    """F.flash_attention agrees with attention in float64, on
    the CPU (as traced) and on MPS and CUDA (one flash-attention kernel)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    (q, k, v), arrays = tensors(device, dtype, (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv))

    def f(q, k, v):
        """Flash attention, causal when the case is."""
        return F.flash_attention(q, k, v, is_causal=causal)

    out = lumen.compile(f)(q, k, v)
    assert out.dtype == dtype and list(out.shape) == [b, sq, n, hv]
    np.testing.assert_allclose(
        lumen.to_numpy(out.to(dtype="float32")), reference(*arrays, causal=causal), rtol=TOL[dtype], atol=TOL[dtype]
    )
    if device in ("mps", "cuda"):
        assert steps(f, q, k, v, device=device) == ["flash_attention"]


DECODE_CASES = [
    (1, 1, 64, 32, 8, 64),
    (2, 4, 100, 4, 2, 32),
    (1, 16, 72, 4, 4, 64),
]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "device, case",
    [("cpu", c) for c in DECODE_CASES]
    + [pytest.param("mps", c, marks=pytest.mark.mps) for c in DECODE_CASES]
    + [pytest.param("cuda", c, marks=pytest.mark.cuda) for c in DECODE_CASES],
    ids=[f"cpu-{c}" for c in DECODE_CASES] + [f"mps-{c}" for c in DECODE_CASES] + [f"cuda-{c}" for c in DECODE_CASES],
)
def test_attention_decode(device, case, dtype, triton_on_cuda):
    """Decode attention over a KV cache agrees with causal attention over its first `start + Sq` rows,
    for several cache positions, and is one kernel on MPS and CUDA."""
    b, sq, sk, n, nkv, h = case
    (q, k, v), arrays = tensors(device, dtype, (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, h))
    q_, k_, v_ = arrays

    def f(q, k, v, start):
        """Decode attention of `q` at cache position `start`."""
        return F.attention_decode(q, k, v, start)

    step = lumen.compile(f)
    for start in (0, sk // 2, sk - sq):
        pos = lumen.tensor(start, dtype=lumen.int32, device=device)
        out = step(q, k, v, pos)
        assert out.dtype == dtype and list(out.shape) == [b, sq, n, h]
        seen = start + sq
        want = reference(q_, k_[:, :seen], v_[:, :seen], causal=True)
        np.testing.assert_allclose(lumen.to_numpy(out.to(dtype="float32")), want, rtol=TOL[dtype], atol=TOL[dtype])
    if device in ("mps", "cuda"):
        assert steps(f, q, k, v, pos, device=device) == ["flash_attention"]


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_attention_as_traced_without_flash_attention(device, triton_on_cuda):
    """With ``flash_attention`` off, attention runs as traced (scores in
    memory, several kernels); with it on (the default), one kernel,
    agreeing to rounding."""
    (q, k, v), _ = tensors(device, "float16", (1, 64, 4, 32), (1, 64, 2, 32), (1, 64, 2, 32))

    def f(q, k, v):
        """Causal flash attention."""
        return F.flash_attention(q, k, v, is_causal=True)

    config = lumen.config.compiler.of(device)
    assert config.flash_attention is True
    assert steps(f, q, k, v, device=device) == ["flash_attention"]
    flash = lumen.to_numpy(lumen.compile(f)(q, k, v).to(dtype="float32").to("cpu"))
    try:
        config.flash_attention = False
        labels = steps(f, q, k, v, device=device)
        assert len(labels) > 1 and "flash_attention" not in labels
        traced = lumen.to_numpy(lumen.compile(f)(q, k, v).to(dtype="float32").to("cpu"))
    finally:
        config.flash_attention = True
    np.testing.assert_allclose(flash, traced, rtol=4e-3, atol=4e-3)


def _written(q, k, v):
    """Attention as a model writes it: [B, N, S, H], grouped-query heads by
    expanding K, a scale, the softmax in float32 and cast back."""
    b, n, s, h = q.shape
    nkv = k.shape[1]
    k = k.unsqueeze(2).expand(b, nkv, n // nkv, s, h).reshape(b, n, s, h)
    scores = F.matmul(q, k.transpose(-1, -2), "float32", q.dtype) * (1.0 / h**0.5)
    return F.matmul(F.softmax(scores.float(), -1).to(dtype=q.dtype), v, "float32", q.dtype)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_attention_written_out_is_one_kernel(device, triton_on_cuda):
    """Attention written out (``_written``) is matched too: one kernel,
    agreeing with the CPU; at a model's size too."""
    (q, k, v), arrays = tensors(device, "bfloat16", (2, 4, 24, 16), (2, 2, 24, 16), (2, 4, 24, 16))
    assert steps(_written, q, k, v, device=device) == ["flash_attention"]
    got = lumen.to_numpy(lumen.compile(_written)(q, k, v).to(dtype="float32").to("cpu"))
    cpu = [lumen.from_numpy(a).to(dtype="bfloat16") for a in arrays]
    want = lumen.to_numpy(lumen.compile(_written, device="cpu")(*cpu).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=3e-2, atol=3e-2)
    big = [
        lumen.empty(s, dtype="bfloat16", device="meta")
        for s in ([2, 32, 1024, 128], [2, 8, 1024, 128], [2, 32, 1024, 128])
    ]
    assert steps(_written, *big, device=device) == ["flash_attention"]


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_attention_the_kernels_cannot_take_runs_as_traced(device, triton_on_cuda):
    """A head dimension past the kernels' (256) or not a multiple of 8 runs
    as traced, still correct."""
    for h in (512, 20):
        (q, k, v), arrays = tensors(device, "float32", (1, 16, 2, h), (1, 16, 2, h), (1, 16, 2, h))

        def f(q, k, v):
            """Non-causal flash attention."""
            return F.flash_attention(q, k, v)

        assert "flash_attention" not in steps(f, q, k, v, device=device)
        out = lumen.to_numpy(lumen.compile(f)(q, k, v).to("cpu"))
        np.testing.assert_allclose(out, reference(*arrays), rtol=1e-5, atol=1e-5)


def test_flash_attention_checks_its_operands():
    """Flash attention rejects mismatched head counts, ranks and dtypes, and integer operands."""

    def call(q, k, v):
        """Traces flash attention on `q`, `k`, `v`."""
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


@pytest.mark.parametrize("device", [MPS, CUDA])
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
def test_attention_epilogues_fuse(f, sq, device, triton_on_cuda):
    """The elementwise primitives after an attention, through reshapes (a
    residual, a bias, a scale, a cast), run in its kernel as it writes each
    output (on MPS, ``contraction_epilogues``): one step, labeled with them,
    agreeing with the CPU and, on MPS, with the unfused plan."""
    shapes = [(1, sq, 4, 32), (1, 96, 4, 32), (1, 96, 4, 32), (sq, 128), (128,)]
    ts, arrays = tensors(device, "float32", *shapes)
    got = lumen.to_numpy(lumen.compile(f)(*ts).to(dtype="float32").to("cpu"))
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(*(lumen.from_numpy(a) for a in arrays)).to(dtype="float32"))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    (label,) = steps(f, *ts, device=device)
    assert label.startswith("flash_attention → reshape → "), label
    if device != "mps":
        return
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        assert len(steps(f, *ts)) == 2
        unfused = lumen.to_numpy(lumen.compile(f)(*ts).to(dtype="float32"))
    finally:
        lumen.config.compiler.reset()
    np.testing.assert_allclose(got, unfused, rtol=1e-6, atol=1e-6)


def _sources(f, *args):
    """Generated kernel source of each step of `f` planned for MPS, by step label."""
    plan = lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "mps")
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
        lambda q, k, v: F.matmul(
            F.softmax(F.matmul(q.half(), k.half().transpose(-1, -2), "float32", "float16"), -1).to(dtype=q.dtype),
            v,
            "float32",
            q.dtype,
        ),
        lambda q, k, v: F.matmul(
            F.softmax(
                prims.dot_general(q, k, (((3,), (3,)), ((0, 1), (0, 1))), "bfloat16", "bfloat16").float(), -1
            ).to(dtype=q.dtype),
            v,
            "float32",
            q.dtype,
        ),
    ],
    ids=["f16 softmax", "bf16 accumulation"],
)
def test_attention_the_kernel_cannot_compute_as_traced_runs_as_traced(f):
    """Attention computing in a precision other than the kernel's (float32
    softmax and accumulation) is not matched: it runs as traced."""
    m = [lumen.empty([1, 4, 64, 32], dtype="bfloat16", device="meta")] * 3
    assert "flash_attention" not in _sources(f, *m)


BACKWARD_CASES = [
    ((1, 40, 40, 4, 2, 32, 32), True),
    ((1, 70, 20, 2, 2, 16, 16), False),
    ((1, 3, 50, 2, 1, 16, 24), True),
    ((1, 9, 24, 2, 2, 128, 128), True),
]


def backward_kernels(h, hv, dtype, deterministic=False, device="mps"):
    """An attention backward's kernels: one adding dQ atomically unless
    ``deterministic`` (on MPS, if its blocks fit in threadgroup memory too:
    codegen.rs's ``backward_block_with_dq``); else two."""
    t = {"float32": 4, "bfloat16": 2}[dtype]
    fits = device == "cuda" or any(n * (h + hv) * t + 8 * n + 64 * h * t + 64 * n * t <= 28 << 10 for n in (32, 16))
    if fits and not deterministic:
        return ["flash_attention_backward(dk, dv, dq)"]
    return ["flash_attention_backward(dk, dv)", "flash_attention_backward(dq)"]


@pytest.mark.parametrize("deterministic", [False, True], ids=["", "deterministic"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "device, case",
    [pytest.param("mps", c, marks=pytest.mark.mps) for c in BACKWARD_CASES]
    + [pytest.param("cuda", c, marks=pytest.mark.cuda) for c in BACKWARD_CASES],
    ids=[f"mps-{i}" for i in _ids(BACKWARD_CASES)] + [f"cuda-{i}" for i in _ids(BACKWARD_CASES)],
)
def test_attention_backward_is_flash_attention(device, case, dtype, deterministic, triton_on_cuda):
    """The gradient of F.flash_attention: the forward one flash-attention
    kernel (writing each row's log-sum-exp too), the backward one adding dQ
    atomically by its dK and dV kernel (on MPS where its blocks fit), or
    two (dK and dV, then dQ; ``deterministic``: the same bits each run),
    recomputing the probabilities; agreeing with the CPU's (the same
    program, as traced)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    shapes = (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv), (b, sq, n, hv)
    (q, k, v, w), arrays = tensors(device, dtype, *shapes)

    def loss(q, k, v, w):
        """A weighted sum of attention's output, so every output element gets a distinct gradient."""
        return F.sum((F.flash_attention(q, k, v, is_causal=causal) * w).float())

    grad = lumen.grad(loss, (0, 1, 2))
    config = lumen.config.compiler.of(device)
    config.deterministic = deterministic
    try:
        labels = steps(grad, q, k, v, w, device=device)
        for kernel in ["flash_attention", *backward_kernels(h, hv, dtype, deterministic, device)]:
            assert kernel in labels, labels
        compiled = lumen.compile(grad)
        got = [t.to("cpu") for t in compiled(q, k, v, w)]
        if deterministic:
            for g, again in zip(got, compiled(q, k, v, w)):
                g, again = (lumen.to_numpy(t.to(dtype="float32").to("cpu")) for t in (g, again))
                np.testing.assert_array_equal(g, again)
    finally:
        config.deterministic = False
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
        """A weighted sum of attention's output with dropout."""
        return F.sum((attention(q, k, v, is_causal=causal, dropout_p=0.25) * w).float())

    step = lumen.value_and_grad(loss, (0, 1, 2))
    lumen.config.compiler.mps.deterministic = deterministic
    try:
        labels = steps(step, q, k, v, w)
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
        """Sum of the written-out attention."""
        return F.sum(_written(q, k, v).float())

    labels = steps(lumen.grad(loss, (0, 1, 2)), *m)
    for kernel in ["flash_attention", *backward_kernels(32, 32, "bfloat16")]:
        assert kernel in labels, labels


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("scale", [1.0, 0.125], ids=["unscaled", "scaled"])
def test_written_attention_backward_is_flash_attention(scale, device, triton_on_cuda):
    """Attention written out trains with flash attention's backward kernels
    whether its scores are scaled or not (``* 1`` simplified away, dS
    has no scale to multiply by: 1); the gradients the CPU's."""
    (q, k, v), arrays = tensors(device, "float32", (2, 40, 32), (2, 40, 32), (2, 40, 32))

    def loss(q, k, v):
        """Sum of tanh of a written-out single-head attention with optionally scaled scores."""
        return F.sum(
            F.tanh(
                F.matmul(
                    F.softmax(F.matmul(q, k.transpose(-1, -2), "float32", "float32") * scale, -1),
                    v,
                    "float32",
                    "float32",
                )
            )
        )

    grad = lumen.grad(loss, (0, 1, 2))
    labels = steps(grad, q, k, v, device=device)
    kernels = [label.split(" → ")[0] for label in labels]
    for kernel in ["flash_attention", *backward_kernels(32, 32, "float32", device=device)]:
        assert kernel in kernels, labels
    got = lumen.compile(grad)(q, k, v)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), lumen.to_numpy(w), rtol=1e-4, atol=1e-5)


def _strided(device, layout, b, s, n, h, seed=0):
    """A `[b, s, n, h]` view of a larger or reordered buffer, its head dim contiguous, laid out as `layout`
    names (`fused` a third of a fused QKV projection's heads, `heads first` a `[b, n, s, h]` buffer
    transposed, `padded` rows of a wider head dim at an offset, `window` a window of longer sequences), with its
    values as a float32 array."""
    shape = {
        "fused": (b, s, 3 * n, h),
        "heads first": (b, n, s, h),
        "padded": (b, s, n, h + 8),
        "window": (b, s + 8, n, h),
    }[layout]
    (t,), (a,) = tensors(device, "float32", shape, seed=seed)
    view, values = {
        "fused": lambda: (t.narrow(2, n, n), a[:, :, n : 2 * n]),
        "heads first": lambda: (t.transpose(1, 2), a.transpose(0, 2, 1, 3)),
        "padded": lambda: (t.narrow(3, 4, h), a[..., 4 : 4 + h]),
        "window": lambda: (t.narrow(1, 4, s), a[:, 4 : 4 + s]),
    }[layout]()
    assert not view.is_contiguous()
    return view, np.ascontiguousarray(values)


LAYOUTS = [
    ("fused", "fused", "fused"),
    ("heads first", "heads first", "heads first"),
    ("padded", "padded", "padded"),
    ("window", "window", "window"),
    ("heads first", "padded", "fused"),
]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("layouts", LAYOUTS, ids=["/".join(layout) for layout in LAYOUTS])
def test_attention_reads_strided_operands_in_place(device, layouts, triton_on_cuda):
    """Q, K and V at any strides but a contiguous head dim (a fused
    projection's heads, heads-first data transposed, padded rows at an
    offset, a window of longer sequences, or each its own) are read where they are,
    forward and backward: no copy first; the attention and its gradients
    the CPU's."""
    b, s, n, h = 2, 40, 4, 32
    (q, a_q), (k, a_k), (v, a_v) = (_strided(device, layout, b, s, n, h, seed=i) for i, layout in enumerate(layouts))
    arrays = [a_q, a_k, a_v]
    (w,), (wa,) = tensors(device, "float32", (b, s, n, h), seed=9)

    def loss(q, k, v, w):
        """A weighted sum of causal attention's output."""
        return F.sum(F.flash_attention(q, k, v, is_causal=True) * w)

    attend = lumen.compile(lambda q, k, v: F.flash_attention(q, k, v, is_causal=True))
    step = lumen.compile(lumen.value_and_grad(loss, (0, 1, 2)))
    attend(q, k, v), step(q, k, v, w)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = attend(q, k, v)
        value, grads = step(q, k, v, w)
    # No copy of an operand: neither a contiguous one first nor a buffer for one in an attention step.
    events = {e["id"]: e for e in prof.events()}
    attention = lambda e: e.get("parent") in events and events[e["parent"]]["name"].startswith("flash_attention")
    assert not [e for e in prof.events() if e["name"] == "lumen::contiguous"]
    assert not [e for e in prof.events() if e["name"] == "lumen::empty" and attention(e)]
    np.testing.assert_allclose(lumen.to_numpy(out), reference(*arrays, causal=True), rtol=1e-4, atol=1e-4)
    cpu = [lumen.from_numpy(np.ascontiguousarray(a)) for a in (*arrays, wa)]
    want_value, want_grads = lumen.compile(lumen.value_and_grad(loss, (0, 1, 2)), device="cpu")(*cpu)
    for got, want in zip((value, *grads), (want_value, *want_grads)):
        np.testing.assert_allclose(lumen.to_numpy(got.to("cpu")), lumen.to_numpy(want), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_attention_copies_operands_with_a_strided_head_dim(device, triton_on_cuda):
    """An operand whose head dim isn't contiguous (`[b, s, h, n]` data
    passed transposed) is copied first, as before: the attention the CPU's."""
    b, s, n, h = 1, 24, 2, 16
    ts, arrays = tensors(device, "float32", *[(b, s, h, n)] * 3)
    q, k, v = (t.transpose(2, 3) for t in ts)
    out = lumen.compile(lambda q, k, v: F.flash_attention(q, k, v))(q, k, v)
    want = reference(*(a.transpose(0, 1, 3, 2) for a in arrays))
    np.testing.assert_allclose(lumen.to_numpy(out), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_trained_attention_is_written_through_its_transpose(device, triton_on_cuda):
    """Attention written out heads first (``[b, n, s, h]``), then
    transposed back and its heads merged for the next projection, in a
    training step: its kernel writes ``[b, s, n, h]`` directly (one
    ``flash_attention → reshape``, no transpose kernel), which the backward
    reads through strides; the loss and gradients the CPU's."""
    b, s, n, h = 2, 32, 4, 16
    shapes = [(b * s, n * h), (n * h, n * h), (n * h, n * h), (n * h, n * h), (n * h, n * h)]
    ts, arrays = tensors(device, "float32", *shapes)

    def loss(x, wq, wk, wv, wo):
        """A causal multi-head attention block, heads moved ahead of the sequence, summed after `wo`."""
        q, k, v = (F.matmul(x, w, "float32", "float32").reshape(b, s, n, h).transpose(1, 2) for w in (wq, wk, wv))
        a = F.softmax(F.matmul(q, k.transpose(-1, -2), "float32", "float32") * h**-0.5, dim=-1)
        o = F.matmul(a, v, "float32", "float32").transpose(1, 2).reshape(b * s, n * h)
        return F.sum(F.tanh(F.matmul(o, wo, "float32", "float32")))

    step = lumen.value_and_grad(loss, (0, 1, 2, 3, 4))
    labels = steps(step, *ts, device=device)
    assert any(label.startswith("flash_attention → reshape") for label in labels), labels
    assert "transpose → reshape" not in labels, labels
    value, grads = lumen.compile(step)(*ts)
    want_value, want_grads = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for got, want in zip((value, *grads), (want_value, *want_grads)):
        want = lumen.to_numpy(want)
        np.testing.assert_allclose(lumen.to_numpy(got.to("cpu")), want, rtol=1e-3, atol=1e-3 * np.abs(want).max())


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
    lumen.config.compiler.mps.deterministic = True
    shapes = [(1, sq, 4, 32), (1, 40, 2, 32), (1, 40, 2, 32), (sq, 128)]
    ts, _ = tensors("mps", "float32", *shapes)

    def block(q, k, v, x):
        """Sum of tanh of a residual added to the attention output."""
        return F.sum(F.tanh(x + F.flash_attention(q, k, v, is_causal=True).reshape(sq, 128)))

    step = lumen.value_and_grad(block, (0, 1, 2, 3))
    labels = steps(step, *ts)
    assert "flash_attention → reshape → add → tanh" in labels, labels
    assert "flash_attention_backward(dk, dv)" in labels and "flash_attention_backward(dq)" in labels
    loss, grads = lumen.compile(step)(*ts)
    got = [lumen.to_numpy(t) for t in (loss, *grads)]
    lumen.config.compiler.mps.contraction_epilogues = False
    try:
        loss, grads = lumen.compile(step)(*ts)
    finally:
        lumen.config.compiler.reset()
    for g, w in zip(got, (loss, *grads)):
        np.testing.assert_array_equal(g, lumen.to_numpy(w))


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("sq", [40, 1], ids=["tiled", "decode"])
def test_attention_epilogue_fuses_in_training_on_cuda(sq, device, triton_on_cuda):
    """On CUDA too, a training step's attention runs its epilogue (a
    residual) in its kernel, which writes the attention's value for the
    backward too; the step agrees with the CPU."""
    shapes = [(1, sq, 4, 32), (1, 40, 2, 32), (1, 40, 2, 32), (sq, 128)]
    ts, arrays = tensors(device, "float32", *shapes)

    def block(q, k, v, x):
        """Sum of tanh of a residual added to the attention output."""
        return F.sum(F.tanh(x + F.flash_attention(q, k, v, is_causal=True).reshape(sq, 128)))

    step = lumen.value_and_grad(block, (0, 1, 2, 3))
    labels = steps(step, *ts, device=device)
    assert "flash_attention → reshape → add → tanh" in labels, labels
    loss, grads = lumen.compile(step)(*ts)
    want_loss, want_grads = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in arrays))
    for g, w in zip((loss, *grads), (want_loss, *want_grads)):
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), lumen.to_numpy(w), rtol=1e-3, atol=1e-3)


@pytest.mark.mps
def test_log_sum_exp_is_written_only_for_training():
    """The forward kernel writes each row's log-sum-exp only when a backward
    reads it: without one, it is pruned before compiling."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([1, 64, 4, 32], [1, 64, 2, 32], [1, 64, 2, 32])]

    def forward(q, k, v):
        """Causal flash attention."""
        return F.flash_attention(q, k, v, is_causal=True)

    def source(f):
        """Kernel source of the flash attention step in `f`'s MPS plan."""
        plan = lumen.graph_compiler.Plan(lumen.make_graph(f)(*m), "mps")
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
        """Sum of causal flash attention."""
        return F.sum(F.flash_attention(q, k, v, is_causal=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lumen.make_graph(lumen.grad(loss, (0, 1, 2)))(*m)
