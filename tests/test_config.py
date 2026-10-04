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
    """Every program runs as traced, but for online softmax and flash
    attention (on)."""
    assert (compiler.fuse, compiler.merge_dots, compiler.normalization_diamonds) == (True, True, True)
    assert (compiler.reduction_epilogues, compiler.multi_output_fusion) == (True, True)
    assert compiler.contraction_epilogues is True
    assert compiler.online_softmax is True and compiler.flash_attention is True and compiler.row_cache == 8
    assert repr(compiler).startswith("lumen.config.compiler(fuse=True, merge_dots=True")
    assert "online_softmax=True" in repr(compiler)
    # Every instance reads and writes the same flags.
    compiler.online_softmax, compiler.row_cache = False, 4
    assert not lumen.config.compiler.online_softmax and lumen.config.compiler.row_cache == 4
    compiler.reset()
    assert lumen.config.compiler.online_softmax and lumen.config.compiler.row_cache == 8
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
        (kernel,) = [e["kernel"] for e in prof.events() if e["kind"] == "gpu"]
        return out, kernel

    online, online_kernel = run()
    compiler.online_softmax = False
    exact, exact_kernel = run()
    # Compiled again: another kernel (its name is its source's hash).
    assert online_kernel != exact_kernel
    e = np.exp(x - x.max(-1, keepdims=True))
    np.testing.assert_allclose(exact, e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(online, exact, rtol=1e-5, atol=1e-7)
