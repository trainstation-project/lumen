"""lumen.config: runtime settings shared with the Rust core."""

import numpy as np
import pytest

import lumen
import lumen.functional as F


@pytest.fixture(autouse=True)
def restore_static_allocator_bytes():
    before = lumen.config.static_allocator_bytes
    yield
    lumen.config.static_allocator_bytes = before


def test_static_allocator_bytes_defaults_to_one_gib():
    assert lumen.config.static_allocator_bytes == 1 << 30


def test_static_allocator_bytes_can_be_set():
    lumen.config.static_allocator_bytes = 4 << 20
    assert lumen.config.static_allocator_bytes == 4 << 20
    assert repr(lumen.config) == f"lumen.config(static_allocator_bytes={4 << 20})"


def test_static_allocator_bytes_requires_a_non_negative_int():
    with pytest.raises(TypeError):
        lumen.config.static_allocator_bytes = "1GiB"
    with pytest.raises(OverflowError):
        lumen.config.static_allocator_bytes = -1


def test_unknown_settings_are_rejected():
    with pytest.raises(AttributeError):
        lumen.config.memory_caching = False  # the setting no longer exists


def test_config_is_a_single_shared_instance():
    assert lumen.config is lumen._C.config
    assert isinstance(lumen.config, lumen._C.Config)


# ---------------------------------------------------------------------
# lumen.config.compiler
# ---------------------------------------------------------------------


@pytest.fixture
def compiler():
    """``lumen.config.compiler``, reset to its defaults after the test."""
    yield lumen.config.compiler
    lumen.config.compiler.reset()


def _manual_softmax(a):
    e = F.exp(a - F.amax(a, -1, keepdim=True))
    return e / F.sum(e, -1, keepdim=True)


def _labels(f, *args, device="mps"):
    try:
        return [s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(*args), device).steps()]
    except RuntimeError as e:
        pytest.skip(str(e))


def test_compiler_flags_defaults(compiler):
    """Every program runs as traced, but for online softmax, flash
    attention and split-K (on)."""
    assert (compiler.fuse, compiler.merge_dots, compiler.normalization_diamonds) == (True, True, True)
    assert (compiler.reduction_epilogues, compiler.multi_output_fusion) == (True, True)
    assert compiler.contraction_epilogues is True and compiler.horizontal_fusion is True
    assert compiler.online_softmax is True and compiler.flash_attention is True and compiler.row_cache == 8
    assert compiler.split_k is True and compiler.deterministic is False and compiler.memory_limit == 0
    assert compiler.fused_linear_cross_entropy_chunk_size is None
    assert repr(compiler).startswith("lumen.config.compiler(fuse=True, merge_dots=True")
    assert "online_softmax=True" in repr(compiler)
    # Every instance reads and writes the same flags.
    compiler.online_softmax, compiler.row_cache = False, 4
    assert not lumen.config.compiler.online_softmax and lumen.config.compiler.row_cache == 4
    compiler.reset()
    assert lumen.config.compiler.online_softmax and lumen.config.compiler.row_cache == 8
    compiler.fused_linear_cross_entropy_chunk_size = 128
    assert "fused_linear_cross_entropy_chunk_size=128" in repr(compiler)
    compiler.fused_linear_cross_entropy_chunk_size = None
    assert "fused_linear_cross_entropy_chunk_size=None" in repr(compiler)
    with pytest.raises(TypeError):
        compiler.fuse = "yes"
    with pytest.raises(OverflowError):
        compiler.row_cache = -1
    with pytest.raises(AttributeError):
        compiler.fast = True


@pytest.mark.mps
def test_compiler_flags_change_what_compiles(compiler):
    """Each flag turns its pass or kernel shape on or off (on MPS)."""
    x = lumen.empty([64, 300], device="meta")
    assert len(_labels(_manual_softmax, x)) == 1
    compiler.normalization_diamonds = False
    assert len(_labels(_manual_softmax, x)) > 1
    compiler.reset()

    cast = lambda a: F.sum(a.float(), -1).half()  # noqa: E731
    h = lumen.empty([64, 300], dtype="float16", device="meta")
    assert len(_labels(cast, h)) == 1
    compiler.reduction_epilogues = False
    assert len(_labels(cast, h)) == 2
    compiler.reset()

    compiler.fuse = False
    assert "fusion" not in " ".join(_labels(_manual_softmax, x)) and len(_labels(_manual_softmax, x)) > 3
    compiler.reset()

    big = lumen.empty([8, 65536], device="meta")
    online = lambda: lumen.graph.Plan(lumen.make_graph(_manual_softmax)(big), "mps").steps()[0]  # noqa: E731
    assert "maxima" in online()["fusion"]["source"]
    compiler.online_softmax = False
    assert "maxima" not in online()["fusion"]["source"]
    compiler.reset()

    (step,) = lumen.graph.Plan(lumen.make_graph(_manual_softmax)(x), "mps").steps()
    assert "kept0[" in step["fusion"]["source"]
    compiler.row_cache = 0
    (step,) = lumen.graph.Plan(lumen.make_graph(_manual_softmax)(x), "mps").steps()
    assert "kept0[" not in step["fusion"]["source"]
    compiler.reset()

    apart = lambda a, b: (F.exp(a) + 1.0, b * 2.0 - 1.0)  # noqa: E731
    assert len(_labels(apart, x, x)) == 1
    compiler.horizontal_fusion = False
    assert len(_labels(apart, x, x)) == 2


def _read_early_and_late(a, w):
    h = a @ w
    b = h @ w
    e = (b @ w) @ b
    return F.sum((e @ w) * h, -1)


@pytest.mark.mps
def test_memory_limit_recomputes(compiler):
    """Under ``memory_limit``, a value read early and late is computed again
    for its late reader rather than kept alive (rematerialization): ``h``,
    alive with ``b``, ``b @ w`` and ``e`` where four values are, is
    recomputed for the last matmul: a matmul more, a value less of
    workspace, the same results."""
    a, w = lumen.empty([512, 512], device="meta"), lumen.empty([512, 512], device="meta")

    def plan():
        try:
            return lumen.graph.Plan(lumen.make_graph(_read_early_and_late)(a, w), "mps")
        except RuntimeError as e:
            pytest.skip(str(e))

    def dots(p):
        return sum("dot_general" in s["label"] for s in p.steps())

    kept = plan()
    compiler.memory_limit = 1
    recomputed = plan()
    assert recomputed.workspace_bytes == kept.workspace_bytes - 512 * 512 * 4
    assert dots(recomputed) == dots(kept) + 1
    rng = np.random.default_rng(0)
    a, w = (lumen.from_numpy(rng.standard_normal((512, 512)).astype(np.float32) / 16).to("mps") for _ in range(2))
    got = lumen.to_numpy(lumen.compile(_read_early_and_late)(a, w))
    compiler.reset()
    want = lumen.to_numpy(lumen.compile(_read_early_and_late)(a, w))
    np.testing.assert_array_equal(got, want)


@pytest.mark.mps
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_split_k_agrees(compiler, dtype):
    """With ``split_k`` (the default) a matmul launching few threadgroups and
    of a long contraction (a decode step's) is one dot of the chunks'
    partials, then their sum (each labelled split-K), rounded to bfloat16
    once, after it: atomic adds allowed (not ``deterministic``) or not, its
    partials cost less than its operands. A larger output's (partials
    costing more) is one kernel adding its chunks' products to its float32
    output atomically, unless ``deterministic``. Each agrees with the dot
    as traced (off) to rounding. One of many tiles is not split."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 4096)).astype(np.float32), rng.standard_normal((4096, 256)).astype(np.float32)
    try:
        x, w = (lumen.from_numpy(t).to("mps").to(dtype=dtype) for t in (a, b))
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lambda x, w: x @ w  # noqa: E731
    want = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    cast = "cast(float32 -> bfloat16)"
    rounded = {"float32": "", "bfloat16": " → " + cast}[dtype]
    assert _labels(f, x, w) == ["dot_general (split-K)", "reduce_sum (split-K)" + rounded]
    split = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
    compiler.deterministic = True
    assert _labels(f, x, w) == ["dot_general (split-K)", "reduce_sum (split-K)" + rounded]
    ordered = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
    compiler.split_k = False
    assert _labels(f, x, w) == ["dot_general" + rounded]
    traced = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
    tol = {"float32": 1e-4, "bfloat16": 2e-2}[dtype] * np.abs(want).max()
    for got in (split, ordered, traced):
        np.testing.assert_allclose(got, want, atol=tol)
    compiler.reset()
    big = lumen.empty([1024, 1024], device="meta")
    assert _labels(f, big, big) == ["dot_general"]
    # A larger output: atomically (in bfloat16, its partials in float32).
    a, b = (rng.standard_normal(s).astype(np.float32) for s in ((352, 2048), (2048, 352)))
    x, w = (lumen.from_numpy(t).to("mps").to(dtype="bfloat16") for t in (a, b))
    want = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    assert _labels(f, x, w) == ["dot_general (split-K)", cast]
    atomic = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
    np.testing.assert_allclose(atomic, want, atol=2e-2 * np.abs(want).max())
    compiler.deterministic = True
    assert _labels(f, x, w) == ["dot_general (split-K)", "reduce_sum (split-K) → " + cast]


@pytest.mark.mps
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_split_k_then_silu(compiler, dtype):
    """SiLU after a matmul of few output tiles and a long contraction: not
    split, it is the matmul's epilogue (one kernel); split, it needs every
    chunk's products: the epilogue of the ``reduce_sum`` of the dot's
    partials (reading the sum twice), atomic adds allowed or not (its
    partials cost less than its operands). Each agrees with SiLU of the
    exact product to rounding."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((4, 4096)).astype(np.float32), rng.standard_normal((4096, 256)).astype(np.float32)
    try:
        x, w = (lumen.from_numpy(t).to("mps").to(dtype=dtype) for t in (a, b))
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(x, w):
        y = F.matmul(x, w, accum_dtype=lumen.float32, output_dtype=lumen.float32)
        return (y * F.sigmoid(y)).to(dtype=x.dtype)

    y = lumen.to_numpy(x.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(w.to(dtype="float32"))
    want = y / (1 + np.exp(-y))
    silu = "logistic → mul" + {"float32": "", "bfloat16": " → cast(float32 -> bfloat16)"}[dtype]
    plans = {
        "unsplit": ["dot_general → " + silu],
        "atomic": ["dot_general (split-K)", "reduce_sum (split-K) → " + silu],
        "deterministic": ["dot_general (split-K)", "reduce_sum (split-K) → " + silu],
    }
    tol = {"float32": 1e-4, "bfloat16": 2e-2}[dtype] * np.abs(want).max()
    for mode, plan in plans.items():
        compiler.reset()
        compiler.split_k = mode != "unsplit"
        compiler.deterministic = mode == "deterministic"
        assert _labels(f, x, w) == plan, mode
        got = lumen.to_numpy(lumen.compile(f)(x, w).to(dtype="float32"))
        np.testing.assert_allclose(got, want, atol=tol, err_msg=mode)


@pytest.mark.mps
@pytest.mark.parametrize("n", [300, 65536])
def test_online_softmax_agrees(compiler, n):
    """With ``online_softmax`` (the default) a softmax's max and sum are one
    pass; it agrees with softmax as traced (off) to rounding.
    ``lumen.compile`` compiles again when a flag changes."""
    x = np.random.default_rng(0).standard_normal((8, n)).astype(np.float32) * 3
    try:
        t = lumen.from_numpy(x).to("mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    from lumen.profiler import ProfilerActivity, profile

    f = lumen.compile(lambda a: F.softmax(a, -1))

    def run():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            out = lumen.to_numpy(f(t))
        (kernel,) = [e["kernel"] for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
        return out, kernel

    online, online_kernel = run()
    compiler.online_softmax = False
    exact, exact_kernel = run()
    # Compiled again: another kernel (its name is its source's hash).
    assert online_kernel != exact_kernel
    e = np.exp(x - x.max(-1, keepdims=True))
    np.testing.assert_allclose(exact, e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(online, exact, rtol=1e-5, atol=1e-7)
