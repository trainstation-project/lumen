"""Attention: F.scaled_dot_product_attention ([B, S, N, H], grouped-query,
causal), and the MPS compiler running attention, however written, as one
flash-attention kernel (lumen/compiler/mps/attention.rs, attention.metal)."""

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
def test_scaled_dot_product_attention(device, case, dtype):
    """F.scaled_dot_product_attention agrees with attention in float64, on
    the CPU (as traced) and on MPS (one flash-attention kernel)."""
    (b, sq, sk, n, nkv, h, hv), causal = case
    (q, k, v), arrays = tensors(device, dtype, (b, sq, n, h), (b, sk, nkv, h), (b, sk, nkv, hv))

    def f(q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

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
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
            return F.scaled_dot_product_attention(q, k, v)

        assert "flash_attention" not in steps(f, q, k, v)
        out = lumen.to_numpy(lumen.compile(f)(q, k, v))
        np.testing.assert_allclose(out, reference(*arrays), rtol=1e-5, atol=1e-5)


def test_scaled_dot_product_attention_checks_its_operands():
    def call(q, k, v):
        return lumen.make_graph(F.scaled_dot_product_attention)(q, k, v)

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
        (lambda q, k, v, x, b: x + F.scaled_dot_product_attention(q, k, v).reshape(-1, 128), 96),
        (
            lambda q, k, v, x, b: (
                F.scaled_dot_product_attention(q, k, v, is_causal=True).reshape(-1, 128) * 2.0 + b
            ).to(dtype="float16"),
            96,
        ),
        (lambda q, k, v, x, b: x + F.scaled_dot_product_attention(q, k, v, is_causal=True).reshape(-1, 128), 1),
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
    F.scaled_dot_product_attention's in float32."""
    m = [lumen.empty(s, dtype="bfloat16", device="meta") for s in ([1, 4, 64, 32], [1, 2, 64, 32], [1, 4, 64, 32])]
    source = _sources(_written, *m)["flash_attention"]
    assert "convert_value<float>(Mul::apply(convert_value<bfloat>(s), as_type<bfloat>" in source, source
    m = [lumen.empty([1, 64, 4, 32], dtype="bfloat16", device="meta")] * 3
    source = _sources(F.scaled_dot_product_attention, *m)["flash_attention"]
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
