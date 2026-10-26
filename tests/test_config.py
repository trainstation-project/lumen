"""Tests of the per-device compiler settings in `lumen.config.compiler` and the passes each flag controls."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen.config._C import CompilerConfig

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)


@pytest.fixture
def compiler():
    """``lumen.config.compiler.mps``, every device's settings reset to their
    defaults after the test."""
    yield lumen.config.compiler.mps
    lumen.config.compiler.reset()


def _manual_softmax(a):
    """Softmax over the last axis written out of primitives."""
    e = F.exp(a - F.amax(a, -1, keepdim=True))
    return e / F.sum(e, -1, keepdim=True)


def _labels(f, *args, device="mps"):
    """Step labels of `f` traced on `args` and planned for `device`, skipping when it is unavailable."""
    try:
        return [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), device).steps()]
    except RuntimeError as e:
        pytest.skip(str(e))


def test_compiler_flags_defaults(compiler):
    """Each device's compiler has its own settings (``lumen.config.compiler``
    holds one per device; ``of`` picks a device's, the meta device's the
    CPU's): every program runs as traced, but for online softmax, flash
    attention, fused linear cross entropy and split-K (on). A device's
    settings hold only the flags its compiler reads, the ones every device
    has from their base class (``CompilerConfig``)."""
    configs = lumen.config.compiler
    for device in (configs.cpu, configs.cuda, configs.mps, configs.tpu):
        assert (device.merge_dots, device.flash_attention, device.memory_limit) == (True, True, 0)
        assert device.fused_linear_cross_entropy_chunk_size is None and device.dump_to is None
    assert (compiler.fuse, compiler.normalization_diamonds) == (True, True)
    assert (compiler.reduction_epilogues, compiler.multi_output_fusion) == (True, True)
    assert compiler.contraction_epilogues is True and compiler.horizontal_fusion is True
    assert compiler.online_softmax is True and compiler.row_cache == 8
    assert compiler.split_k is True and compiler.deterministic is False
    assert compiler.fused_linear_cross_entropy is True and compiler.safe_kernels is False
    assert compiler.autotune is True and compiler.neural_engine is False
    assert (configs.cuda.use_triton_backend, configs.cuda.autotune) == (False, True)
    tpu = configs.tpu
    assert (tpu.normalization_diamonds, tpu.contraction_epilogues, tpu.horizontal_fusion) == (True, True, True)
    assert tpu.safe_kernels is False
    assert repr(compiler).startswith("lumen.config.compiler.mps(merge_dots=True, flash_attention=True")
    assert all(isinstance(c, CompilerConfig) for c in (configs.cpu, configs.cuda, compiler, tpu))
    assert "online_softmax=True" in repr(compiler)
    assert type(configs.of("mps")).__name__ == "MpsConfig" and type(configs.of("meta")).__name__ == "CpuConfig"
    compiler.online_softmax, compiler.row_cache = False, 4
    assert not configs.of("mps").online_softmax and configs.mps.row_cache == 4
    configs.cpu.merge_dots = False
    assert configs.mps.merge_dots and not configs.of("cpu").merge_dots
    compiler.reset()
    assert configs.mps.online_softmax and configs.mps.row_cache == 8 and not configs.cpu.merge_dots
    configs.reset()
    assert configs.cpu.merge_dots
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
    with pytest.raises(AttributeError):
        configs.cpu.split_k = False


def test_stub_defaults_are_the_compilers(compiler):
    """The defaults ``lumen/config/_C.pyi`` shows for each device's flags
    (``split_k: bool = True``) are the ones its compiler starts from: each
    device's flags its own and its base class's (``CompilerConfig``), every
    flag given one."""
    import ast
    import pathlib

    stub = ast.parse((pathlib.Path(lumen.__file__).parent / "config" / "_C.pyi").read_text())
    classes = {c.name: c for c in stub.body if isinstance(c, ast.ClassDef)}

    def flags(name):
        """Flag defaults the stub declares for class `name`, its base class's included."""
        fields = {a.target.id: ast.literal_eval(a.value) for a in classes[name].body if isinstance(a, ast.AnnAssign)}
        return {**(flags(classes[name].bases[0].id) if classes[name].bases else {}), **fields}

    lumen.config.compiler.reset()
    for device in ("cpu", "cuda", "mps", "tpu"):
        config = getattr(lumen.config.compiler, device)
        documented = flags(type(config).__name__)
        actual = {k for k in dir(config) if not k.startswith("_") and k != "reset"}
        assert set(documented) == actual, device
        assert {k: getattr(config, k) for k in documented} == documented, device


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
    online = lambda: lumen.graph_compiler.Plan(lumen.make_graph(_manual_softmax)(big), "mps").steps()[0]  # noqa: E731
    assert "maxima" in online()["fusion"]["source"]
    compiler.online_softmax = False
    assert "maxima" not in online()["fusion"]["source"]
    compiler.reset()

    take = lambda x, i: x[i] * 2.0  # noqa: E731
    table, ids = lumen.empty([16, 8], device="meta"), lumen.empty([4], dtype="int64", device="meta")
    assert "select" not in " ".join(_labels(take, table, ids))
    compiler.safe_kernels = True
    assert "select" in " ".join(_labels(take, table, ids))
    compiler.reset()

    loss = lambda h, w, t: F.cross_entropy(F.matmul(h, w.t(), "float32", "float32"), t)  # noqa: E731
    args = (lumen.empty([64, 32], device="meta"), lumen.empty([1000, 32], device="meta"))
    args += (lumen.empty([64], dtype="int64", device="meta"),)
    assert "linear_cross_entropy" in _labels(loss, *args)
    compiler.fused_linear_cross_entropy = False
    assert "linear_cross_entropy" not in _labels(loss, *args)
    compiler.reset()

    (step,) = lumen.graph_compiler.Plan(lumen.make_graph(_manual_softmax)(x), "mps").steps()
    assert "kept0[" in step["fusion"]["source"]
    compiler.row_cache = 0
    (step,) = lumen.graph_compiler.Plan(lumen.make_graph(_manual_softmax)(x), "mps").steps()
    assert "kept0[" not in step["fusion"]["source"]
    compiler.reset()

    apart = lambda a, b: (F.exp(a) + 1.0, b * 2.0 - 1.0)  # noqa: E731
    assert len(_labels(apart, x, x)) == 1
    compiler.horizontal_fusion = False
    assert len(_labels(apart, x, x)) == 2


def _read_early_and_late(a, w):
    """A chain of matmuls reading `h` early and again at the end, so it stays alive across the chain."""
    h = F.matmul(a, w, "float32", "float32")
    b = F.matmul(h, w, "float32", "float32")
    e = F.matmul(F.matmul(b, w, "float32", "float32"), b, "float32", "float32")
    return F.sum(F.matmul(e, w, "float32", "float32") * h, -1)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_dump_to_writes_each_pass(compiler, tmp_path, device, triton_on_cuda):
    """``dump_to``: each compiled graph written before its passes and after
    each, a file each, named by its pipeline, step and pass (XLA's
    ``--xla_dump_to``); the MPS pipeline's last passes generate the
    kernels, then place the slices read in place; the Triton pipeline's
    fuses loops."""
    f = lambda x: F.exp(x) * 2.0 + 1.0  # noqa: E731
    graph = lumen.make_graph(f)(lumen.empty([8], device="meta"))
    lumen.config.compiler.cpu.dump_to = lumen.config.compiler.of(device).dump_to = str(tmp_path)
    available = getattr(lumen, device).is_available()
    devices = ["cpu"] + ([device] if available else [])
    for d in devices:
        lumen.graph_compiler.Plan(graph, d)
    steps = {p.name.split(".", 1)[1] for p in tmp_path.iterdir()}
    assert {"frontend.00.input.txt", "frontend.01.cse.txt", "frontend.02.simplify.txt"} <= steps
    assert "cpu.01.simplify.txt" in steps
    if "mps" in devices:
        mps = sorted(s for s in steps if s.startswith("mps."))
        assert mps[-2].endswith(".codegen.txt") and mps[-1].endswith(".place_views.txt"), mps
        assert any(".loop_fusion." in s for s in mps)
        assert (
            "fusion_" in (tmp_path / next(p.name for p in tmp_path.iterdir() if p.name.endswith(mps[-2]))).read_text()
        )
    if "cuda" in devices:
        assert any(s.startswith("triton.") and ".loop_fusion." in s for s in steps), steps
    lumen.config.compiler.reset()
    count = len(list(tmp_path.iterdir()))
    lumen.graph_compiler.Plan(graph, "cpu")
    assert len(list(tmp_path.iterdir())) == count


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_memory_limit_recomputes(compiler, device, triton_on_cuda):
    """Under ``memory_limit``, a value read early and late is computed again
    for its late reader rather than kept alive (rematerialization): ``h``,
    alive with ``b``, ``b @ w`` and ``e`` where four values are, is
    recomputed for the last matmul: a matmul more, a value less of
    workspace, the same results."""
    a, w = lumen.empty([512, 512], device="meta"), lumen.empty([512, 512], device="meta")

    def plan():
        """Plans the chain for `device`, skipping when it is unavailable."""
        try:
            return lumen.graph_compiler.Plan(lumen.make_graph(_read_early_and_late)(a, w), device)
        except RuntimeError as e:
            pytest.skip(str(e))

    def dots(p):
        """Number of dot steps in plan `p`."""
        return sum("dot_general" in s["label"] for s in p.steps())

    config = lumen.config.compiler.of(device)
    kept = plan()
    config.memory_limit = 1
    recomputed = plan()
    if device == "mps":
        assert recomputed.workspace_bytes == kept.workspace_bytes - 512 * 512 * 4
        assert dots(recomputed) == dots(kept) + 1
    rng = np.random.default_rng(0)
    a, w = (lumen.from_numpy(rng.standard_normal((512, 512)).astype(np.float32) / 16).to(device) for _ in range(2))
    got = lumen.to_numpy(lumen.compile(_read_early_and_late)(a, w).to("cpu"))
    config.memory_limit = 0
    want = lumen.to_numpy(lumen.compile(_read_early_and_late)(a, w).to("cpu"))
    if device == "mps":
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5 * np.abs(want).max())


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
    f = lambda x, w: F.matmul(x, w, "float32", x.dtype)  # noqa: E731
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
        """SiLU of a float32-accumulated matmul, cast back to the input dtype."""
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
        """Runs the softmax under the profiler, returning its output and the one GPU kernel it launched."""
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            out = lumen.to_numpy(f(t))
        (kernel,) = [e["kernel"] for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
        return out, kernel

    online, online_kernel = run()
    compiler.online_softmax = False
    exact, exact_kernel = run()
    assert online_kernel != exact_kernel
    e = np.exp(x - x.max(-1, keepdims=True))
    np.testing.assert_allclose(exact, e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(online, exact, rtol=1e-5, atol=1e-7)
