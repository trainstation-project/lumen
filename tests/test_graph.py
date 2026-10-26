"""Tests for tracing, graphs, plans and compiled programs: semantics, fusion, devices, modules and training steps."""

import dataclasses
import re
import warnings

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims
from lumen.profiler import ProfilerActivity, profile

MPS = pytest.param("mps", marks=pytest.mark.mps)
CUDA = pytest.param("cuda", marks=pytest.mark.cuda)
TPU = pytest.param("tpu", marks=pytest.mark.tpu)


def run(fn, *args):
    """``lumen.compile(fn)(*args)`` with the results as NumPy arrays."""
    out = lumen.compile(fn)(*args)
    return tuple(map(lumen.to_numpy, out)) if isinstance(out, tuple) else lumen.to_numpy(out)


def rand(*shape, dtype=np.float32, seed=0):
    """Standard normal array of `shape` from a seeded generator."""
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def test_graph_text():
    """A traced graph prints as its primitives with types and parameters."""
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x), -1))(lumen.zeros([2, 3]))
    assert str(graph) == (
        "{ lambda %0:f32[2,3]. let\n"
        "    %1:f32[2,3] = exp %0\n"
        "    %2:f32[2] = reduce_sum[axes=(1,) accum_dtype=f32] %1\n"
        "  in (%2) }"
    )


def test_traces_once_per_signature():
    """A compiled function retraces only for new shapes, dtypes or Python scalar values."""
    traces = []

    @lumen.compile
    def f(x, scale):
        """Records each trace's input shape and scales `x`."""
        traces.append(x.shape)
        return x * scale

    f(lumen.ones([2]), 2.0)
    f(lumen.ones([2]), 2.0)
    assert len(traces) == 1
    assert lumen.to_numpy(f(lumen.ones([3]), 2.0)).tolist() == [2.0] * 3
    assert lumen.to_numpy(f(lumen.ones([3]), 3.0)).tolist() == [3.0] * 3
    assert len(traces) == 2
    assert f(lumen.ones([3], dtype="float64"), 3.0).dtype == "float64"
    assert len(traces) == 3
    f(lumen.ones([3]), 2)
    f(lumen.ones([3]), 3)
    assert len(traces) == 5


def test_multiple_outputs_and_passthrough():
    """A program may return an input unchanged alongside computed values."""
    x = lumen.tensor([1.0, 2.0])
    a, b = run(lambda x: (x, x + 1), x)
    assert a.tolist() == [1.0, 2.0] and b.tolist() == [2.0, 3.0]


def test_ops_need_a_trace():
    """Primitives outside a trace raise, and eager tensor arithmetic is not supported."""
    with pytest.raises(RuntimeError, match="while tracing"):
        prims.full((2,), 1.0, "float32")
    with pytest.raises(TypeError):
        lumen.tensor([1.0]) + 1


def test_data_dependent_control_flow_is_rejected():
    """Branching on a traced value raises a TypeError."""

    def f(x):
        """Branches on the sign of the sum of `x`."""
        return x if F.sum(x) > 0 else -x

    with pytest.raises(TypeError, match="control flow"):
        lumen.compile(f)(lumen.ones([2]))


def test_must_return_traced_tensors():
    """A program returning a non-tensor raises a TypeError."""
    with pytest.raises(TypeError):
        lumen.compile(lambda x: 1.0)(lumen.ones([2]))


def test_factories_record_primitives_while_tracing():
    """Factories called while tracing become graph nodes broadcast like operands."""

    def f(x):
        """Adds tensors from each factory to `x`."""
        return x + lumen.ones([3]) + lumen.full([2, 3], 2.0) + lumen.arange(3) + lumen.zeros([3])

    x = lumen.zeros([2, 3])
    assert "iota" in str(lumen.make_graph(f)(x))
    out = run(f, x)
    assert out.dtype == np.float32
    assert out.tolist() == [[3.0, 4.0, 5.0]] * 2
    assert run(lambda: lumen.full([2], True)).tolist() == [True, True]
    assert isinstance(lumen.ones([2]), lumen.Tensor)


def test_prims_do_not_broadcast_or_promote():
    """Primitives need operands of one type and float ops float operands."""
    x, y = lumen.zeros([2, 3]), lumen.zeros([3])
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, y)
    with pytest.raises(ValueError, match="same type"):
        lumen.compile(prims.add)(x, lumen.zeros([2, 3], dtype="float64"))
    with pytest.raises(ValueError, match="floating-point"):
        lumen.compile(prims.exp)(lumen.zeros([2], dtype="int32"))


def test_prims_compose():
    """Primitives compose into a program numpy agrees with."""
    x = rand(2, 3)

    def f(x):
        """Where `x` is below its row max, `x` plus its column index, else the row max."""
        m = prims.broadcast_in_dim(prims.reduce_max(x, (1,)), (2, 3), (0,))
        i = prims.iota("float32", (2, 3), 1)
        return prims.select(prims.lt(x, m), prims.add(x, i), m)

    m = x.max(1, keepdims=True)
    np.testing.assert_array_equal(run(f, lumen.from_numpy(x)), np.where(x < m, x + np.arange(3, dtype=np.float32), m))


def test_no_implicit_dtype_changes():
    """Python scalars take the tensor's dtype, and mixed dtypes raise instead of promoting."""
    i, f = lumen.tensor([1, 2]), lumen.tensor([1.0, 2.0])
    half = lumen.tensor([1.0], dtype="float16")
    assert run(lambda i: i * 2, i).dtype == np.int64
    assert run(lambda f: f * 2, f).dtype == np.float32
    assert run(lambda h: h + 2.5, half).dtype == np.float16
    assert run(lambda b: b == True, lumen.tensor([True, False])).tolist() == [True, False]  # noqa: E712
    u8 = lumen.tensor([200, 100], dtype="uint8")
    assert run(lambda x: F.sum(x), u8).tolist() == 44
    assert run(lambda x: F.sum(x.long()), u8).tolist() == 300
    assert run(lambda i: F.exp(i.float()), i).dtype == np.float32
    assert run(lambda i, f: i.to(f.dtype) + f, i, f).tolist() == [2.0, 4.0]
    assert run(lambda i: F.softmax(i.float(), 0), i).dtype == np.float32
    errors = [
        (lambda i, f: i + f, (i, f), "dtypes float32 and int64"),
        (lambda h, f: F.maximum(h, f), (half, lumen.tensor([0.0])), "dtypes float16 and float32"),
        (lambda f, h: F.where(f > 0, f, h), (lumen.tensor([1.0]), half), "dtypes float16 and float32"),
        (
            lambda h, f: F.matmul(h, f, "float32", "float16"),
            (lumen.zeros([2, 2], dtype="float16"), lumen.zeros([2, 2])),
            "dtypes float16 and float32",
        ),
        (lambda i: i + 1.5, (i,), "int64 tensor and the float 1.5"),
        (lambda b: b + 1, (lumen.tensor([True]),), "bool tensor and the int 1"),
        (lambda i: i / 2, (i,), "true division"),
        (lambda i: F.exp(i), (i,), "exp needs a floating-point tensor"),
        (lambda i: F.softmax(i, 0), (i,), "softmax needs a floating-point tensor"),
    ]
    for fn, args, message in errors:
        with pytest.raises(TypeError, match=message):
            lumen.compile(fn)(*args)


def test_broadcasting():
    """Operators broadcast as numpy does, and incompatible shapes raise."""
    a, b = rand(4, 1, 3), rand(5, 1, seed=1)
    np.testing.assert_allclose(
        run(lambda a, b: a * b - a, lumen.from_numpy(a), lumen.from_numpy(b)), a * b - a, rtol=1e-6
    )
    with pytest.raises(RuntimeError, match="broadcastable"):
        lumen.compile(lambda a, b: a + b)(lumen.zeros([2]), lumen.zeros([3]))


def test_a_matmul_names_its_dtypes():
    """`@` and an einsum without dtypes raise, asking for them."""
    x = lumen.empty([2, 3], device="meta")
    with pytest.raises(TypeError, match="@ has no dtypes"):
        lumen.make_graph(lambda x: x @ x.t())(x)
    with pytest.raises(TypeError, match="accum_dtype"):
        lumen.make_graph(lambda x: F.einsum("mk,nk->mn", x, x))(x)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [((3,), (3,)), ((2, 3), (3,)), ((3,), (3, 4)), ((2, 3), (3, 4)), ((5, 2, 3), (3, 4)), ((1, 2, 3), (4, 3, 2))],
)
def test_matmul(a_shape, b_shape):
    """F.matmul of vectors, matrices and batches matches numpy."""
    a, b = rand(*a_shape), rand(*b_shape, seed=1)
    out = run(lambda a, b: F.matmul(a, b, "float32", a.dtype), lumen.from_numpy(a), lumen.from_numpy(b))
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-6)


def floats(t):
    """`t` on the host as float32 numpy."""
    return lumen.to_numpy(t.to("cpu").to(dtype="float32"))


def placed(shape, dtype, device, seed=0):
    """A seeded standard normal tensor of `shape` as `dtype` on `device`."""
    a = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32)
    return lumen.from_numpy(a).to(dtype=dtype).to(device)


@pytest.mark.parametrize("device", ["cpu", MPS, TPU, CUDA])
@pytest.mark.parametrize(
    "lhs, rhs, dtype, transposed, output",
    [
        ([256, 512], [512, 384], "float32", False, None),
        ([64, 1024], [1024, 128], "float32", False, None),
        ([1, 256], [256, 256], "float32", False, None),
        ([2, 3, 256], [256, 128], "float32", False, None),
        ([2, 64, 256], [2, 256, 384], "float32", False, None),
        ([4, 8], [8, 16], "float32", False, None),
        ([3, 100], [100, 30], "bfloat16", False, None),
        ([64, 256], [384, 256], "float32", True, None),
        ([8, 1024], [512, 1024], "bfloat16", True, None),
        ([8, 1024], [512, 1024], "bfloat16", True, "float32"),
    ],
)
def test_dots_agree_with_numpy(device, lhs, rhs, dtype, transposed, output, triton_on_cuda):
    """A dot, in float32 on the operands as placed, within its operands'
    dtype's precision of its largest output (float32's when it writes
    float32)."""
    x, w = placed(lhs, dtype, device), placed(rhs, dtype, device, seed=1)
    out = output or dtype
    y = lumen.compile(lambda x, w: F.matmul(x, w.t() if transposed else w, "float32", out))(x, w)
    assert y.device == f"{device}:0" and y.dtype == out
    a, b = floats(x), floats(w)
    expected = a @ (b.T if transposed else b)
    assert list(y.shape) == list(expected.shape)
    tol = 1e-5 if out == "float32" else 1e-2
    assert np.abs(floats(y) - expected).max() <= tol * np.abs(expected).max()


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_triton_loops_match_the_cpu(device, dtype, triton_on_cuda):
    """With ``use_triton_backend``, each loop fusion on CUDA (elementwise
    ops, broadcasts, transposes, slices, iota and constants) is one generated
    Triton kernel computing what the CPU does; bfloat16's functions are
    float32's, cast as the program asks."""
    a, b = rand(48, 80), rand(80, 48, seed=1)

    def wide(f):
        """`f` computed in float32 and cast back to the input's dtype."""
        return lambda v: f(v.to(dtype="float32")).to(dtype=v.dtype)

    exp, tanh, sqrt, sigmoid, log = map(wide, (F.exp, F.tanh, F.sqrt, F.sigmoid, F.log))
    fns = [
        lambda x, y: exp(x) * 2.0 + tanh(y.t()),
        lambda x, y: F.where(x > y.t(), sqrt(F.relu(x)), sigmoid(-x)) / (F.maximum(x, y.t()) + 3.0),
        lambda x, y: log(x * x + 1.0)[:, 8:40] - y.t()[:, :32] + prims.iota(x.dtype, (48, 32), 1),
        lambda x, y: (x < y.t(), (x * 4.0).to(dtype="int32")),
    ]
    tol = {"float32": 1e-5, "bfloat16": 2**-7}[dtype]
    for f in fns:
        on_cpu = [lumen.from_numpy(v).to(dtype=dtype) for v in (a, b)]
        try:
            on_device = [v.to(device) for v in on_cpu]
        except RuntimeError as e:
            pytest.skip(str(e))
        expected, got = lumen.compile(f)(*on_cpu), lumen.compile(f)(*on_device)
        for e, g in zip(*(r if isinstance(r, tuple) else (r,) for r in (expected, got))):
            assert g.device == f"{device}:0" and g.dtype == e.dtype
            e, g = (lumen.to_numpy(r.to(dtype="float32")) for r in (e, g))
            np.testing.assert_allclose(g, e, rtol=tol, atol=tol)
    plan = lumen.graph_compiler.Plan(lumen.make_graph(fns[1])(*on_device), device)
    assert len(plan.steps()) == 1
    scaled = lumen.compile(lambda x, s: x * s + 1.0)
    for s in (0.5, -3.0):
        out = lumen.to_numpy(scaled(on_device[0], s).to(dtype="float32"))
        np.testing.assert_allclose(out, lumen.to_numpy(scaled(on_cpu[0], s).to(dtype="float32")), rtol=tol)


def triton_agrees(device, f, arrays, tol):
    """`f` compiled on `device` and on the CPU gives the same dtype and values within `tol` (exactly when 0)."""
    on_cpu = [lumen.from_numpy(np.ascontiguousarray(v)) for v in arrays]
    try:
        on_device = [v.to(device) for v in on_cpu]
    except RuntimeError as e:
        pytest.skip(str(e))
    expected, got = lumen.compile(f)(*on_cpu), lumen.compile(f)(*on_device)
    assert got.device == f"{device}:0" and got.dtype == expected.dtype
    e, g = (
        lumen.to_numpy(r.to(dtype="float32") if r.dtype in ("bfloat16", "float16") else r) for r in (expected, got)
    )
    if tol:
        np.testing.assert_allclose(g, e, rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(g, e)


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("shape", [(37, 300), (300, 37), (5, 70, 9), (1, 1), (2, 0, 3)])
def test_triton_reductions_match_the_cpu(device, shape, triton_on_cuda):
    """With ``use_triton_backend``, sums and maxima over any axes (rows,
    columns, middle, several, all; lengths no tile divides, empty ones)
    are one Triton kernel each, accumulating in the reduction's dtype:
    float32 and bfloat16, integers wrapping, NaN propagating through max."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    x.flat[::7] = np.nan
    ints = rng.integers(-100, 100, shape).astype(np.int8)
    axes_list = [(d,) for d in range(len(shape))] + [tuple(range(len(shape)))]
    if len(shape) == 3:
        axes_list += [(0, 2)]
    bf16 = lambda v: v.to(dtype="bfloat16")  # noqa: E731
    for axes in axes_list:
        triton_agrees(device, lambda v: prims.reduce_max(v, axes), [x], 0)
        triton_agrees(device, lambda v: prims.reduce_sum(v, axes, "float32"), [np.nan_to_num(x)], 1e-5)
        triton_agrees(device, lambda v: prims.reduce_sum(bf16(v), axes, "float32"), [np.nan_to_num(x)], 1e-5)
        triton_agrees(device, lambda v: prims.reduce_sum(bf16(v), axes, "bfloat16"), [np.nan_to_num(x)], 2e-1)
        triton_agrees(device, lambda v: prims.reduce_sum(v, axes, "int8"), [ints], 0)
        triton_agrees(device, lambda v: prims.reduce_max(v, axes), [ints], 0)


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("shape, axes", [((3_000_001,), (0,)), ((7, 200_003), (1,)), ((90_001, 5), (0,))])
def test_triton_long_reductions_split_and_match_the_cpu(device, shape, axes, triton_on_cuda):
    """A reduction to few outputs over many elements runs in two passes,
    partial results in the reduction's dtype and then their combination and
    epilogue: the CPU's sum, mean and max, integers exactly."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    ints = rng.integers(-100, 100, shape).astype(np.int32)
    triton_agrees(device, lambda v: prims.reduce_sum(v, axes, "float32"), [x], 1e-3)
    triton_agrees(device, lambda v: F.mean(F.exp(v), axes[0]) * 2.0, [x], 1e-3)
    triton_agrees(device, lambda v: prims.reduce_max(v, axes).to(dtype="bfloat16"), [x], 0)
    triton_agrees(device, lambda v: prims.reduce_sum(v, axes, "int32"), [ints], 0)


@pytest.mark.parametrize("device", [CUDA])
def test_triton_reductions_fuse_their_prologue_and_epilogue(device, triton_on_cuda):
    """The elementwise ops feeding a reduction and those applied to its
    result run in the reduction's kernel: one step, the CPU's value."""
    a, b = rand(64, 96), rand(96, seed=1)
    fns = [
        lambda x, y: F.sum(F.exp(x) * 2.0 - y, -1) * 0.5 + 1.0,
        lambda x, y: F.amax(x.t() + y.reshape(96, 1), 0).to(dtype="bfloat16"),
        lambda x, y: F.mean(x * x, 1) + y[:64],
    ]
    for f in fns:
        triton_agrees(device, f, [a, b], 1e-5)
        plan = lumen.graph_compiler.Plan(
            lumen.make_graph(f)(*(lumen.from_numpy(v).to(device) for v in (a, b))), device
        )
        assert len(plan.steps()) == 1, [s["label"] for s in plan.steps()]


@pytest.mark.parametrize("device", [CUDA])
def test_triton_matmul_epilogues_fuse(device, triton_on_cuda):
    """The elementwise ops applied to a matmul's result (a bias, an
    activation, a transposed input, a cast, a reshape, a batched matmul's)
    run in its GEMM's epilogue: one step, the CPU's values."""
    a, w, b, r = rand(64, 96), rand(96, 80, seed=1), rand(80, seed=2), rand(80, 64, seed=3)
    batched = rand(2, 32, 96, seed=4)

    def mm(x, y):
        """float32 matmul."""
        return F.matmul(x, y, "float32", "float32")

    fns = [
        (lambda x, y, z, t: F.relu(mm(x, y) + z), [a, w, b, r]),
        (lambda x, y, z, t: (mm(x, y) * 0.5 + t.t()).to(dtype="bfloat16"), [a, w, b, r]),
        (lambda x, y, z, t: mm(x, y).reshape(8, 8, 80) - z, [a, w, b, r]),
        (lambda x, y, z, t: F.tanh(mm(x, y) + z), [batched, w, b, r]),
    ]
    for f, arrays in fns:
        triton_agrees(device, f, arrays, 1e-4)
        plan = lumen.graph_compiler.Plan(
            lumen.make_graph(f)(*(lumen.from_numpy(v).to(device) for v in arrays)), device
        )
        assert len(plan.steps()) == 1, [s["label"] for s in plan.steps()]
    x, y = (lumen.from_numpy(v).to(device) for v in (a, w))
    scaled = lumen.compile(lambda x, y, s: mm(x, y) * s)
    for s in (0.5, -3.0):
        np.testing.assert_allclose(lumen.to_numpy(scaled(x, y, s).to("cpu")), (a @ w) * s, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", [CUDA])
def test_triton_softmax_and_rms_norm_match_the_cpu(device, triton_on_cuda):
    """Softmax, log-softmax and RMS norm, traced as reductions and loops,
    run on Triton as the CPU computes them."""
    x, w = rand(33, 1000), rand(1000, seed=1)
    triton_agrees(device, lambda v: F.softmax(v, -1), [x], 1e-6)
    triton_agrees(device, lambda v: F.log_softmax(v, 0), [x], 1e-5)
    triton_agrees(device, lambda v, g: F.rms_norm(v, 1000, g), [x, w], 1e-5)


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("shape, axis", [((1000,), 0), ((7, 300), 1), ((300, 7), 0), ((4, 33, 5), 1), ((3, 0), 1)])
def test_triton_cumsum_matches_the_cpu(device, shape, axis, triton_on_cuda):
    """Cumulative sums along any axis, forwards and reversed, on Triton:
    the CPU's, integers exactly, bfloat16 widened to float32 as it is read."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    ints = rng.integers(-100, 100, shape).astype(np.int32)
    for reverse in (False, True):
        triton_agrees(device, lambda v: prims.cumsum(v, axis, reverse, "float32"), [x], 1e-4)
        triton_agrees(device, lambda v: prims.cumsum(v, axis, reverse, "int32"), [ints], 0)
        triton_agrees(device, lambda v: prims.cumsum(v.to(dtype="bfloat16"), axis, reverse, "float32"), [x], 1e-4)


@pytest.mark.parametrize("device", [CUDA])
def test_triton_reads_strided_inputs_in_place(device, triton_on_cuda):
    """With ``use_triton_backend``, a transposed, narrowed or permuted CUDA
    input is read where it is, through a generated load prologue (a loop's
    ``LOAD``, the GEMM's ``PROLOGUE_A``/``PROLOGUE_B``), not copied first;
    ``contiguous`` and ``copy_`` between strided views run on the GPU."""
    import lumen.codegen.triton as triton

    a, b, c = rand(64, 96), rand(96, 64, seed=1), rand(2, 3, 96, seed=2)
    try:
        x, w, z = (lumen.from_numpy(v).to(device) for v in (a, b, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    loop = lumen.compile(lambda u, v: F.exp(u) * 2.0 + v)
    dot = lumen.compile(lambda u, v: F.matmul(u, v, "float32", "float32"))

    def strided(kind, run):
        """Runs `run`, checking a new `kind` kernel indexes its input with strides; returns the result."""
        before = set(triton.kernels)
        out = lumen.to_numpy(run())
        new = [triton.compiled(name, None) for name in set(triton.kernels) - before if name.startswith(kind)]
        assert any("//" in f.src or "*" in f.src for f in new)
        return out

    tol = {"rtol": 1e-5, "atol": 1e-5}
    out = strided("load_", lambda: loop(x.transpose(0, 1), w))
    np.testing.assert_allclose(out, np.exp(a.T) * 2 + b, **tol)
    out = loop(x.narrow(1, 8, 32), w.narrow(0, 8, 64).narrow(1, 0, 32))
    np.testing.assert_allclose(lumen.to_numpy(out), np.exp(a[:, 8:40]) * 2 + b[8:72, :32], **tol)
    out = strided("prologue_", lambda: dot(x.transpose(0, 1), x))
    np.testing.assert_allclose(out, a.T @ a, rtol=1e-4, atol=1e-4)
    out = strided("prologue_", lambda: dot(z.transpose(0, 1), w))
    np.testing.assert_allclose(out, c.transpose(1, 0, 2) @ b, rtol=1e-4, atol=1e-4)
    t = x.transpose(0, 1).contiguous()
    assert t.is_contiguous() and lumen.to_numpy(t).tolist() == a.T.tolist()
    y = lumen.zeros([96, 64], device=device)
    y.transpose(0, 1).copy_(x)
    assert lumen.to_numpy(y).tolist() == a.T.tolist()


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_triton_autotuned_loops_match_untuned_and_dump_their_launch(device, dtype, tmp_path, triton_on_cuda):
    """A Triton loop the compiler autotunes computes what the untuned loop
    does, at lengths no ``BLOCK_SIZE`` divides, and its dumped source carries
    the ``BLOCK_SIZE`` and warps it picked."""
    compiler = lumen.config.compiler.cuda
    try:
        for n in ((1 << 16) + 7, 1 << 20, (3 << 20) + 1):
            try:
                x = lumen.from_numpy(rand(n, seed=n)).to(device).to(dtype=dtype)
            except RuntimeError as e:
                pytest.skip(str(e))
            results = []
            for autotune in (False, True):
                compiler.autotune = autotune
                y = lumen.compile(lambda x: -x * 2)(x)
                results.append(lumen.to_numpy(y.to(dtype="float32").to("cpu")))
            assert np.array_equal(results[0], results[1]), n
        lumen.compile(lambda x: -x * 2).dump_code(str(tmp_path), x)
        source = "".join(p.read_text() for p in tmp_path.glob("*fusion_*.py"))
        assert re.search(r"# BLOCK_SIZE=\d+ warps=\d+", source)
        assert "BLOCK_SIZE: tl.constexpr" in source
    finally:
        compiler.autotune = True


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_mps_reads_strided_inputs_in_place(device, triton_on_cuda):
    """A transposed or narrowed MPS or CUDA input is read where it is, not copied
    first: its step's kernel is compiled again with that input's load
    prologue through its strides, once a layout. ``contiguous``, ``copy_``
    and ``fill_`` of strided views are loops on the GPU, not host copies."""
    a, b = rand(64, 96), rand(96, 64, seed=1)
    x, w = lumen.from_numpy(a).to(device), lumen.from_numpy(b).to(device)
    loop = lumen.compile(lambda u, v: F.exp(u) * 2.0 + v)
    norm = lumen.compile(lambda u: u / F.sqrt(F.mean(u * u, -1, keepdim=True) + 1e-6))
    tol = {"rtol": 1e-5, "atol": 1e-5}
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        out = loop(x.transpose(0, 1), w)
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(a.T) * 2 + b, **tol)
        out = loop(x.narrow(1, 8, 32), w.narrow(0, 8, 64).narrow(1, 0, 32))
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(a[:, 8:40]) * 2 + b[8:72, :32], **tol)
        out = norm(x.transpose(0, 1))
        expected = a.T / np.sqrt((a.T.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6)
        np.testing.assert_allclose(lumen.to_numpy(out), expected, **tol)
    assert not [e for e in prof.events() if e["name"] == "lumen::contiguous"]
    y = lumen.zeros([96, 64], device=device)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        t = x.transpose(0, 1).contiguous()
        y.transpose(0, 1).copy_(x)
        y.transpose(0, 1)[3].fill_(-2.0)
    host = {"lumen::to_vec", "lumen::copy_h2d", "lumen::copy_d2h"}
    assert not [e for e in prof.events() if e["name"] in host]
    assert t.is_contiguous() and lumen.to_numpy(t).tolist() == a.T.tolist()
    expected = a.T.copy()
    expected[:, 3] = -2.0
    assert lumen.to_numpy(y).tolist() == expected.tolist()


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_mps_steps_replay_on_each_calls_buffers(device, triton_on_cuda):
    """An MPS plan's steps are encoded once and replayed after: each later
    call computes on its own inputs and workspace (a strided call between
    them too), as the first did."""
    f = lumen.compile(lambda x, w: F.sum(F.relu(F.matmul(x, w, "float32", "float32")), -1))
    expected = lambda a, b: np.maximum(a @ b, 0).sum(-1)  # noqa: E731
    for seed in range(3):
        a, b = rand(32, 64, seed=seed), rand(64, 48, seed=seed + 10)
        x, w = lumen.from_numpy(a).to(device), lumen.from_numpy(b).to(device)
        np.testing.assert_allclose(lumen.to_numpy(f(x, w)), expected(a, b), rtol=1e-4, atol=1e-4)
        t = lumen.from_numpy(np.ascontiguousarray(b.T)).to(device).transpose(0, 1)
        np.testing.assert_allclose(lumen.to_numpy(f(x, t)), expected(a, b), rtol=1e-4, atol=1e-4)
    a = rand(4096)
    graph = lumen.make_graph(lambda u: F.exp(u) * 2.0 + 1.0)(lumen.from_numpy(a).to(device))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=[0])
    workspaces = [lumen.Tensor.empty([plan.workspace_bytes], "uint8", device) for _ in range(2)]
    for k in range(4):
        v = a + k
        (out,) = plan.run_in(workspaces[k % 2], [lumen.from_numpy(v).to(device)])
        np.testing.assert_allclose(lumen.to_numpy(out), np.exp(v) * 2 + 1, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_a_dot_of_two_vectors_is_a_product_and_a_sum(device, dtype, triton_on_cuda):
    """``rowsum(dO * O)`` as attention's backward traces it, a dot with no
    free dimensions: dot strength reduction's product and row sum, in
    float32."""
    x, y = placed([2, 8, 1024, 128], dtype, device), placed([2, 8, 1024, 128], dtype, device, seed=1)
    out = lumen.compile(lambda x, y: F.einsum("bhsd,bhsd->bhs", x, y, accum_dtype="float32", output_dtype=x.dtype))(
        x, y
    )
    assert out.device == f"{device}:0" and list(out.shape) == [2, 8, 1024]
    expected = (floats(x) * floats(y)).sum(-1)
    assert np.abs(floats(out) - expected).max() <= {"float32": 1e-5, "bfloat16": 1e-2}[dtype] * np.abs(expected).max()


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
@pytest.mark.parametrize("shape", [[64, 128], [2048, 2048], [3, 16, 256, 128], [8, 1024, 1000]])
def test_sums_over_the_last_dimension(device, shape, triton_on_cuda):
    """bfloat16 summed in float32 over its last dimension."""
    x = placed(shape, "bfloat16", device)
    out = lumen.compile(lambda x: F.sum(x.float(), -1))(x)
    assert out.dtype == "float32" and list(out.shape) == shape[:-1]
    np.testing.assert_allclose(floats(out), floats(x).sum(-1), rtol=1e-5, atol=1e-4)


def test_matmul_folds_a_batch_times_a_matrix():
    """A batch times a matrix (``x @ w``, a weight) is one matmul of the
    batch's rows, as torch folds it: the matrix read as it is, never
    broadcast over the batch (a copy); its gradient one matmul too, not a
    sum of the batch's."""
    m = lambda *s: lumen.empty(list(s), device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, w: F.matmul(x, w, "float32", x.dtype))(m(2, 3, 4), m(4, 5))
    assert "broadcast_in_dim" not in str(g) and "f32[6,5] = dot_general" in str(g), g
    grad = lumen.make_graph(lumen.grad(lambda x, w: F.sum(F.matmul(x, w, "float32", x.dtype)), 1))(m(2, 3, 4), m(4, 5))
    assert str(grad).count("dot_general") == 1 and "reduce_sum" not in str(grad), grad


def test_reductions():
    """sum, mean and max over all or some dims match numpy; mean of integers raises."""
    x = rand(2, 3, 4)
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.sum(x), t), x.sum(), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.sum(x, (0, -1), keepdim=True), t), x.sum((0, 2), keepdims=True), rtol=1e-5
    )
    np.testing.assert_allclose(run(lambda x: F.mean(x, 1), t), x.mean(1), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.amax(x, -1), t), x.max(-1))
    np.testing.assert_array_equal(run(lambda x: F.max(x), t), x.max())
    with pytest.raises(RuntimeError, match="floating point"):
        lumen.compile(lambda x: F.mean(x))(lumen.tensor([1, 2]))


def test_softmax_and_friends():
    """softmax, log_softmax, sigmoid and relu match numpy."""
    x = rand(3, 5)
    e = np.exp(x - x.max(-1, keepdims=True))
    t = lumen.from_numpy(x)
    np.testing.assert_allclose(run(lambda x: F.softmax(x, -1), t), e / e.sum(-1, keepdims=True), rtol=1e-5)
    np.testing.assert_allclose(run(lambda x: F.sum(F.softmax(x, dim=0), 0), t), np.ones(5), rtol=1e-5)
    np.testing.assert_allclose(
        run(lambda x: F.log_softmax(x, 1), t), np.log(e / e.sum(-1, keepdims=True)), rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(run(lambda x: F.sigmoid(x), t), 1 / (1 + np.exp(-x)), rtol=1e-5)
    np.testing.assert_array_equal(run(lambda x: F.relu(x), t), np.maximum(x, 0))


def test_comparisons_and_where():
    """Comparison operators, where, minimum and NaN-propagating maximum match numpy."""
    x, y = rand(6), rand(6, seed=1)
    tx, ty = lumen.from_numpy(x), lumen.from_numpy(y)
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        np.testing.assert_array_equal(run(lambda a, b: getattr(a, op)(b), tx, ty), getattr(x, op)(y))
    np.testing.assert_array_equal(run(lambda a, b: F.where(a > b, a, 0.0), tx, ty), np.where(x > y, x, 0.0))
    np.testing.assert_array_equal(run(F.minimum, tx, ty), np.minimum(x, y))
    nan = lumen.tensor([float("nan"), 1.0])
    assert np.isnan(run(F.maximum, nan, lumen.tensor([0.0, 0.0]))[0])


def test_shape_methods():
    """reshape, view, flatten, permute, transpose, mT, squeeze and expand follow numpy; bad shapes and dims raise."""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x.reshape(4, -1): x.reshape(4, -1),
        lambda x: x.view(-1): x.reshape(-1),
        lambda x: x.flatten(1): x.reshape(2, 12),
        lambda x: x.permute(2, 0, 1): x.transpose(2, 0, 1),
        lambda x: x.transpose(0, -1): x.swapaxes(0, -1),
        lambda x: x.mT: x.swapaxes(-2, -1),
        lambda x: x.unsqueeze(1).squeeze(): x,
        lambda x: x.unsqueeze(0).expand(5, -1, -1, -1): np.broadcast_to(x, (5, 2, 3, 4)),
    }
    for fn, expected in cases.items():
        np.testing.assert_array_equal(run(fn, t), expected)
    with pytest.raises(RuntimeError, match="invalid for input of size 24"):
        lumen.compile(lambda x: x.reshape(5, -1))(t)
    with pytest.raises(IndexError, match="Dimension out of range"):
        lumen.compile(lambda x: F.sum(x, 3))(t)


@pytest.mark.parametrize("device", [MPS, CUDA, TPU])
def test_results_on_input_device(device, triton_on_cuda):
    """A compiled program's result lands on its input's device."""
    try:
        x = lumen.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    y = lumen.compile(lambda x: F.matmul(x, x, "float32", x.dtype) + 1)(x)
    assert y.device == str(x.device)
    assert lumen.to_numpy(y).tolist() == [[8.0, 11.0], [16.0, 23.0]]


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_mps_kernels_match_cpu(device, dtype, triton_on_cuda):
    """Compiled functions on MPS (Metal kernels) and CUDA (Triton) agree with the CPU."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def block(x, wq, wk, w1):
        """Attention-like block with a residual, returning a row mean, column max and absolute sum."""
        scores = (
            F.matmul(F.matmul(x, wq, "float32", x.dtype), F.matmul(x, wk, "float32", x.dtype).t(), "float32", x.dtype)
            * 0.25
        )
        h = x + F.matmul(F.softmax(scores.float(), -1).to(dtype=x.dtype), x, "float32", x.dtype)
        return F.mean(F.relu(F.matmul(h, w1, "float32", x.dtype)), -1), F.amax(h, 0), F.sum(F.where(h > 0, h, -h))

    arrays = [rand(8, 16, seed=1), rand(16, 16, seed=2), rand(16, 16, seed=3), rand(16, 32, seed=4)]
    cpu = [lumen.from_numpy(a) for a in arrays]
    fn = lumen.compile(lambda *xs: tuple(o.float() for o in block(*(x.to(dtype) for x in xs))))
    expected = [lumen.to_numpy(o) for o in fn(*cpu)]
    actual = fn(*(t.to(device) for t in cpu))
    tol = {"float32": 1e-4, "float16": 1e-2, "bfloat16": 5e-2}[dtype]
    for e, a in zip(expected, actual):
        assert a.device == f"{device}:0"
        np.testing.assert_allclose(lumen.to_numpy(a), e, rtol=tol, atol=tol)


def test_plan():
    """A plan folds the reshape away, sizes its workspace and checks input dtypes when run."""
    graph = lumen.make_graph(lambda x: F.tanh((F.exp(x) + 1).reshape(-1)))(lumen.zeros([16, 16]))
    plan = lumen.graph_compiler.Plan(graph)
    assert "reshape" not in str(plan)
    assert plan.workspace_bytes == 1024 + 4, plan
    x = rand(16, 16)
    np.testing.assert_allclose(
        lumen.to_numpy(plan.run([lumen.from_numpy(x)])[0]), np.tanh(np.exp(x) + 1).ravel(), rtol=1e-6
    )
    with pytest.raises(ValueError, match="must be f32"):
        plan.run([lumen.zeros([16, 16], dtype="float64")])


def test_graph_and_plan_describe_themselves():
    """`Graph.nodes` and `Plan.steps` describe primitives, inputs and outputs."""
    graph = lumen.make_graph(lambda x: F.sum(F.exp(x * 2.0), -1))(lumen.zeros([2, 3]))
    nodes = graph.nodes()
    assert [n["primitive"] for n in nodes] == ["full", "broadcast_in_dim", "mul", "exp", "reduce_sum"]
    assert nodes[-1]["text"] == "reduce_sum[axes=(1,) accum_dtype=f32]" and nodes[-1]["fusion"] is None
    assert graph.inputs() == [0] and graph.outputs() == [nodes[-1]["output"]]
    steps = lumen.graph_compiler.Plan(graph).steps()
    assert steps[-1]["output"] == ("out0", "float32", [2])
    assert steps[2]["inputs"][0] == ("in0", "float32", [2, 3])


@pytest.mark.mps
def test_mps_plan_steps_and_profiled_kernels():
    """An MPS plan of a row reduction is one fused `reduce_rows` kernel, profiled under its kernel name."""
    try:
        x = lumen.ones([64, 2048], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    from lumen.profiler import ProfilerActivity, profile

    graph = lumen.make_graph(lambda x: F.sum(F.tanh(x * 2.0 + 1.0), -1))(x)
    plan = lumen.graph_compiler.Plan(graph, "mps")
    (step,) = plan.steps()
    fusion = step["fusion"]
    assert fusion["kernel"] in fusion["source"] and "tanh" in fusion["body"] and "reduce_sum" in fusion["body"]
    assert "reduce_rows" in fusion["source"]
    plan.run([x])
    lumen.mps.synchronize()
    with profile(activities=[ProfilerActivity.MPS]) as prof:
        plan.run([x])
        lumen.mps.synchronize()
    kernels = [e["kernel"] for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert kernels == [fusion["kernel"]]
    assert plan.run([x])[0].tolist() == pytest.approx([float(np.tanh(3.0)) * 2048] * 64, rel=1e-5)


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


def test_meta_tensors_have_no_data():
    """Meta tensors have shapes and dtypes but every data read raises."""
    x = lumen.empty([2, 3], device="meta")
    assert (x.device, x.shape, x.dtype) == ("meta:0", [2, 3], "float32")
    assert repr(x) == "Tensor(shape=[2, 3], dtype=f32, device=meta)"
    assert lumen.ones([2, 3], device="meta").device == "meta:0"
    assert lumen.tensor([1.0, 2.0]).to("meta").shape == [2]
    assert x.transpose(0, 1).contiguous().shape == [3, 2]
    for read in (x.tolist, lambda: x.to("cpu"), lambda: x[0, 0], lambda: lumen.to_numpy(x), x.__dlpack__):
        with pytest.raises(RuntimeError, match="no data"):
            read()


def test_tracing_and_running_on_meta_tensors():
    """Programs trace and run on meta tensors, yielding meta results."""

    def f(x, w):
        """relu of a matmul and a row sum."""
        return F.relu(F.matmul(x, w, "float32", x.dtype)), F.sum(x, -1)

    x, w = lumen.empty([8, 4], device="meta"), lumen.empty([4, 16], device="meta")
    assert "dot_general" in str(lumen.make_graph(f)(x, w))
    y, s = lumen.compile(f)(x, w)
    assert (y.device, y.shape, s.shape) == ("meta:0", [8, 16], [8])
    assert lumen.compile(lambda: lumen.arange(5), device="meta")().shape == [5]


def test_compiled_init_runs_once_per_call(tmp_path):
    """An input-less compiled program traces once, and its result is a view of the workspace."""
    calls = []

    def init():
        """Counts its calls and builds a constant."""
        calls.append(1)
        return lumen.full([2, 3], 0.5) + lumen.arange(3)

    make = lumen.compile(init)
    a = make().clone()
    b = make()
    assert calls == [1]
    assert a.tolist() == b.tolist() == [[0.5, 1.5, 2.5]] * 2 and b.shares_storage_with(make())
    data = make.dump_graph(tmp_path / "init.html")
    assert data["inputs"] == [] and data["device"] == "cpu:0"


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_compiled_init_on_device(device, tmp_path, triton_on_cuda):
    """Input-less programs run on the device they are compiled for, and graph dumps report it."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    make = lumen.compile(lambda: (lumen.ones([4, 8]) * 2.0, lumen.arange(8)), device=device)
    w, i = make()
    assert (w.device, i.device) == (f"{device}:0", f"{device}:0")
    assert w.tolist() == [[2.0] * 8] * 4 and i.tolist() == list(map(float, range(8)))
    assert lumen.compile(lambda x: x + 1.0, device=device)(lumen.zeros([2])).tolist() == [1.0, 1.0]
    data = make.dump_graph(tmp_path / "init.html")
    assert data["device"] == f"{device}:0"
    if device == "mps":
        assert any(n.get("kernels") for n in data["views"]["fused"]["nodes"])
    f = lumen.compile(lambda x, w: F.matmul(x, w, "float32", x.dtype))
    meta = lumen.empty([2, 4], device="meta"), lumen.empty([4, 3], device="meta")
    data = f.dump_graph(tmp_path / "f.html", *meta, device=device)
    assert data["device"] == f"{device}:0" and data["inputs"] == ["f32[2,4]", "f32[4,3]"]
    if device == "mps":
        assert any(n.get("kernels") for n in data["views"]["fused"]["nodes"])


class Affine(lumen.nn.Module):
    """Affine layer `x @ w + b`."""

    w: lumen.Tensor
    b: lumen.Tensor

    def __call__(self, x):
        """Applies the layer to `x`."""
        return F.matmul(x, self.w, "float32", x.dtype) + self.b


class Stack(lumen.nn.Module):
    """Submodules in a list, a weight shared by two of them, and a static
    field."""

    layers: list
    scale: float

    def __call__(self, x):
        """Applies each layer then scales."""
        for layer in self.layers:
            x = layer(x) * self.scale
        return x


def meta(*shape):
    """An empty float32 meta tensor of `shape`."""
    return lumen.empty(list(shape), device="meta")


def test_modules_are_frozen_dataclasses_of_weights():
    """Modules list parameters once each, coerce static fields, freeze the rest, and validate state dicts."""
    shared = meta(4)
    m = Stack([Affine(meta(4, 4), shared), Affine(meta(4, 4), shared)], 0.5)
    assert [n for n, _ in m.named_parameters()] == ["layers.0.w", "layers.0.b", "layers.1.w"]
    assert m.parameters()[1] is shared
    m.scale = 1
    assert m.scale == 1.0 and type(m.scale) is float
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.layers = []
    with pytest.raises(TypeError, match="float field"):
        m.scale = "1"
    with pytest.raises(ValueError, match="compile it first"):
        m.load_state_dict({n: lumen.zeros(t.shape) for n, t in m.named_parameters()})
    with pytest.raises(KeyError, match="missing"):
        m.load_state_dict({})


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_module_weights_are_placed_once_and_shared(device, triton_on_cuda):
    """A module argument's tensors are the function's weights: placed
    (zeroed) on the device when it compiles (a call on meta tensors, which
    runs nothing), filled by load_state_dict, shared by every function
    taking the module; each call copies the tensor arguments alone."""
    shared = meta(4)
    model = Stack([Affine(meta(4, 4), shared), Affine(meta(4, 4), shared)], 0.5)
    f = lumen.compile(lambda model, x: model(x), device=device)
    try:
        y = f(model, meta(2, 4))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert (y.device, y.shape) == ("meta:0", [2, 4])
    values = {n: rand(*t.shape, seed=i + 1) for i, (n, t) in enumerate(model.named_parameters())}
    model.load_state_dict({n: lumen.from_numpy(v) for n, v in values.items()})
    x = rand(2, 4)
    h = (x @ values["layers.0.w"] + values["layers.0.b"]) * 0.5
    expected = (h @ values["layers.1.w"] + values["layers.0.b"]) * 0.5
    np.testing.assert_allclose(lumen.to_numpy(f(model, lumen.from_numpy(x))), expected, rtol=1e-5, atol=1e-5)
    g = lumen.compile(lambda layer, x: layer(x), device=device)
    first = model.layers[0]
    np.testing.assert_allclose(lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2, rtol=1e-5, atol=1e-5)
    first.w.copy_(lumen.from_numpy(values["layers.0.w"] + 1))
    np.testing.assert_allclose(
        lumen.to_numpy(g(first, lumen.from_numpy(x))), h * 2 + x.sum(-1, keepdims=True), rtol=1e-5, atol=1e-4
    )
    other = Affine(meta(4, 4), meta(4))
    g(other, meta(2, 4))
    other.load_state_dict({"w": lumen.from_numpy(np.eye(4, dtype=np.float32)), "b": lumen.zeros([4])})
    np.testing.assert_allclose(lumen.to_numpy(g(other, lumen.from_numpy(x))), x, rtol=1e-6)
    with pytest.raises(TypeError, match="cannot close over a tensor"):
        lumen.compile(lambda x: F.matmul(x, first.w, "float32", x.dtype), device=device)(lumen.from_numpy(x))
    with pytest.raises(TypeError, match="whole meta tensors"):
        g(Affine(lumen.zeros([4, 4]), lumen.zeros([4])), lumen.from_numpy(x))


def test_results_are_views_of_the_workspace():
    """Each call's result is a view of the same workspace, overwritten by the next call."""
    f = lumen.compile(lambda x: x * 2.0)
    a = f(lumen.ones([3]))
    assert a.tolist() == [2.0] * 3
    b = f(lumen.full([3], 5.0))
    assert a.shares_storage_with(b) and a.tolist() == [10.0] * 3
    assert not a.clone().shares_storage_with(b)


@pytest.mark.mps
def test_mps_plans_put_copies_and_scratch_in_the_workspace():
    """MPS plans transpose dot operands as separate steps and give large reductions workspace scratch."""
    try:
        lumen.zeros([1], device="mps")
    except RuntimeError as e:
        pytest.skip(str(e))
    dot = lambda a, b: prims.dot_general(a, b, (((0, 2), (2, 0)), ((), ())), a.dtype, a.dtype)  # noqa: E731
    graph = lumen.make_graph(dot)(lumen.zeros([3, 2, 4]), lumen.zeros([4, 5, 3]))
    steps = [s["primitive"] for s in lumen.graph_compiler.Plan(graph, "mps", fuse=False).steps()]
    assert steps == ["transpose", "transpose", "dot_general"]
    graph = lumen.make_graph(lambda x: F.sum(x, -1))(lumen.zeros([4, 50_000]))
    plan = lumen.graph_compiler.Plan(graph, "mps")
    offset, size = plan.steps()[0]["scratch"]
    assert size > 0 and plan.workspace_bytes >= offset + size
    x = lumen.ones([4, 50_000], device="mps")
    assert plan.run([x])[0].tolist() == [50_000.0] * 4


def test_indexing_and_splitting_follow_numpy():
    """Integer, slice, ellipsis and narrow indexing and split/chunk follow numpy; invalid indices raise."""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    t = lumen.from_numpy(x)
    cases = {
        lambda x: x[1]: x[1],
        lambda x: x[:, 1:3]: x[:, 1:3],
        lambda x: x[..., -1]: x[..., -1],
        lambda x: x[0, ..., 1:]: x[0, ..., 1:],
        lambda x: x[-1, -2, -3]: x[-1, -2, -3],
        lambda x: x[:, :, 2:2]: x[:, :, 2:2],
        lambda x: x[:, 5:]: x[:, 5:],
        lambda x: x.narrow(2, 1, 2): x[:, :, 1:3],
        lambda x: x.narrow(-1, -3, 2): x[:, :, 1:3],
    }
    for fn, expected in cases.items():
        out = run(fn, t)
        assert out.shape == expected.shape and np.array_equal(out, expected)
    assert [p.shape for p in lumen.compile(lambda x: x.split(3, 2))(t)] == [[2, 3, 3], [2, 3, 1]]
    assert [p.shape for p in lumen.compile(lambda x: x.split([1, 3], -1))(t)] == [[2, 3, 1], [2, 3, 3]]
    assert [p.shape for p in lumen.compile(lambda x: x.chunk(2, 1))(t)] == [[2, 2, 4], [2, 1, 4]]
    a, b = run(lambda x: x.chunk(2, -1), t)
    assert np.array_equal(a, x[..., :2]) and np.array_equal(b, x[..., 2:])
    errors = [
        (lambda x: x[2], IndexError, "out of bounds"),
        (lambda x: x[0, 0, 0, 0], IndexError, "too many indices"),
        (lambda x: x[..., ...], IndexError, "single ellipsis"),
        (lambda x: x[::2], NotImplementedError, "steps"),
        (lambda x: x[True], TypeError, "indices must be"),
        (lambda x: x.split([1, 1], 2), RuntimeError, "sum exactly to 4"),
        (lambda x: x.narrow(2, 3, 2), IndexError, "narrow"),
    ]
    for fn, error, message in errors:
        with pytest.raises(error, match=message):
            lumen.compile(fn)(t)


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


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_concatenate(device, triton_on_cuda):
    """prims.concatenate joins empty and non-empty operands along a dim and rejects mismatched shapes."""
    a, b, c = rand(2, 3, 4), rand(2, 0, 4, seed=1), rand(2, 5, 4, seed=2)
    try:
        args = [lumen.from_numpy(v).to(device) for v in (a, b, c)]
    except RuntimeError as e:
        pytest.skip(str(e))
    out = lumen.to_numpy(lumen.compile(lambda *xs: prims.concatenate(xs, 1))(*args))
    assert np.array_equal(out, np.concatenate([a, b, c], 1))
    with pytest.raises(ValueError, match="all dimensions but 0"):
        lumen.compile(lambda *xs: prims.concatenate(xs, 0))(*args)


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
    beside the product; its values the CPU's."""
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
                    assert len(steps) == (3 if mixed else 1), [s["label"] for s in steps]
                    (merged,) = [s for s in steps if s["label"].startswith("2x")]
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
        labels = [s["label"] for s in plan.steps() if not s["label"].startswith("cast")]
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
    to interleave read as row halves. The values the unmerged program's."""
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
        if device == "mps":
            graph = lumen.make_graph(step)(make(), meta(*shape))
            plan = lumen.graph_compiler.Plan(graph, "mps", parameters=params, packable=params)
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


@pytest.mark.parametrize("device", [CUDA])
@pytest.mark.parametrize(
    "groups",
    [[1, 1, 1], [2], [4], [8], [2, 1], [8, 4, 2, 1]],
    ids=["ones", "pair", "four", "eight", "pair and one", "eight four two one"],
)
@pytest.mark.parametrize("left", [False, True], ids=["x @ w.T", "w @ x"])
@pytest.mark.parametrize(
    "step", [_gated_inference, _gated_gradients, _gated_training], ids=["inference", "gradients", "training"]
)
def test_triton_dots_sharing_an_operand_merge(device, groups, left, step, triton_on_cuda):
    """On Triton, dots sharing an operand merge into one matmul over their
    weights packed into one block, the weights on either side, forward and
    backward: the values, gradients and trained weights the unmerged
    program's."""
    d, m, h = 32, 24, 64
    n = sum(groups)
    params = list(range(1, n + 1))
    shape = (d, m) if left else (m, d)
    try:
        x = lumen.from_numpy(rand(*shape, seed=9)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def make():
        """A fresh meta model of the case's shapes."""
        return SharedOperand([meta(h, d) for _ in range(n)], groups, left)

    graph = lumen.make_graph(step)(make(), meta(*shape))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=params, packable=params)
    assert sorted(k for members, _ in plan.packed for k in members) == params, plan.packed
    values = {f"ws.{k}": lumen.from_numpy(rand(h, d, seed=k + 1) / 4) for k in range(n)}
    got = []
    try:
        for merge in (True, False):
            lumen.config.compiler.cuda.merge_dots = merge
            model = make()
            f = lumen.compile(step, device=device)
            f(model, meta(*shape))
            model.load_state_dict(values)
            for _ in range(2):
                out = f(model, x)
            out = out if isinstance(out, (list, tuple)) else [out]
            tensors = list(out) + [w._placed(device) for w in model.ws]
            got.append([lumen.to_numpy(t.to("cpu")) for t in tensors])
    finally:
        lumen.config.compiler.cuda.merge_dots = True
    for merged, alone in zip(*got):
        assert np.isfinite(alone).all()
        np.testing.assert_allclose(merged, alone, rtol=1e-4, atol=1e-5 * np.abs(alone).max())


@pytest.mark.parametrize("device", [CUDA])
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
def test_triton_merged_regions_fuse(device, make, n, left, triton_on_cuda):
    """A merged matmul's regions (interleaved groups, plain slices, results
    returned as they are) run in its GEMM's epilogue on Triton, the weights
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
            lumen.config.compiler.cuda.merge_dots = merge
            model = make([meta(h, d) for _ in range(n)], left)
            f = lumen.compile(lambda model, x: model(x), device=device)
            f(model, meta(*shape))
            model.load_state_dict(values)
            out = f(model, x)
            assert all(t.is_contiguous() for t in out)
            got.append([lumen.to_numpy(t.to("cpu")) for t in out])
    finally:
        lumen.config.compiler.cuda.merge_dots = True
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


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_softmax_traces_to_its_primitives(device, triton_on_cuda):
    """softmax traces to its primitives (max, sub, exp, sum, div; no
    primitive of its own): over the last dimension the MPS compiler runs
    them, with the scale before them, as one row kernel (a chain of
    normalization diamonds). Over another dimension too, as kernels."""
    x = rand(8, 300) * 4
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def scaled(t):
        """Softmax of half of `t` over the last dim."""
        return F.softmax(t * 0.5, -1)

    graph = lumen.make_graph(scaled)(t)
    primitives = [n["primitive"] for n in graph.nodes()]
    assert "softmax" not in primitives and {"reduce_max", "exp", "reduce_sum", "div"} <= set(primitives)
    s = x * 0.5
    e = np.exp(s - s.max(-1, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(scaled)(t)), e / e.sum(-1, keepdims=True), rtol=1e-5, atol=1e-7
    )
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(graph, "mps").steps()
        assert step["fusion"] is not None and step["label"].startswith("mul")
    e0 = np.exp(x - x.max(0, keepdims=True))
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(lambda t: F.softmax(t, 0))(t)),
        e0 / e0.sum(0, keepdims=True),
        rtol=1e-5,
        atol=1e-7,
    )


class Norm(lumen.nn.Module):
    """RMS norm through `F.rms_norm`."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` over the weight's shape."""
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_rms_norm(device, triton_on_cuda):
    """F.rms_norm (torch.nn.functional.rms_norm), in a module too: traced
    as primitives, which the MPS compiler recognizes, as it does an RMS norm
    written by hand, and fuses into one kernel (its reduction inside) with
    the ops computing its input."""
    x, w = rand(8, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def expected(x, w, eps):
        """numpy RMS norm of `x`, scaled by `w` if given."""
        y = x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        return y * w if w is not None else y

    f = lambda a, b: F.rms_norm(a + 1.0, 300, b, 1e-6)  # noqa: E731
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(X, W)), expected(x + 1, w, 1e-6), rtol=1e-5, atol=1e-6)
    g = lambda a: F.rms_norm(a, [300])  # noqa: E731  (eps: float32's machine epsilon)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(g)(X)), expected(x, None, 2.0**-23), rtol=1e-5, atol=1e-6)
    assert "rms_norm" not in [n["primitive"] for n in lumen.make_graph(f)(X, W).nodes()]
    by_hand = lambda a, b: b * (a / F.sqrt(1e-6 + F.mean(a * a, -1, keepdim=True)))  # noqa: E731
    np.testing.assert_allclose(
        lumen.to_numpy(lumen.compile(by_hand)(X, W)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6
    )
    if device == "mps":
        steps = lambda f: [
            s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), "mps").steps()
        ]  # noqa: E731
        for g in (f, by_hand):
            (label,) = steps(g)
            assert "reduce_sum" in label and "sqrt" in label, label
        assert steps(f)[0].startswith("add → mul → reduce_sum")
    with pytest.raises(NotImplementedError, match="last dimension"):
        lumen.make_graph(lambda a: F.rms_norm(a, (8, 300)))(X)
    norm = Norm(lumen.empty([300], device="meta"), 1e-6)
    step = lumen.compile(lambda m, a: m(a), device=device)
    step(norm, lumen.empty([8, 300], device="meta"))
    norm.load_state_dict({"weight": W})
    np.testing.assert_allclose(lumen.to_numpy(step(norm, X)), expected(x, w, 1e-6), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", MPS, TPU, CUDA])
def test_sqrt(device, triton_on_cuda):
    """sqrt is a primitive (rsqrt is not: write 1 / F.sqrt(x)); on MPS a fused
    division by a fused sqrt stays a sqrt and a division, as traced."""
    x = rand(8, 16) ** 2 + 0.1
    try:
        t = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    assert not hasattr(F, "rsqrt") and not hasattr(lumen.graph.TracedTensor, "rsqrt")
    for f, expected in [
        (lambda a: F.sqrt(a), np.sqrt(x)),
        (lambda a: 3.0 / F.sqrt(a), 3 / np.sqrt(x)),
    ]:
        np.testing.assert_allclose(lumen.to_numpy(lumen.compile(f)(t)), expected, rtol=1e-6)
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(lambda a: 1.0 / F.sqrt(a))(t), "mps").steps()
        source = step["fusion"]["source"]
        assert step["label"].endswith("div") and "rsqrt(" not in source
        assert "Sqrt::apply" in source and "Div::apply" in source
        f = lambda a: a / F.sqrt(F.sum(a, -1, keepdim=True))  # noqa: E731
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(t), "mps").steps()
        source = step["fusion"]["source"]
        assert "Sqrt::apply" in source and "Div::apply" in source and "rsqrt(" not in source


class ScaledNorm(lumen.nn.Module):
    """RMS norm written out by hand."""

    weight: lumen.Tensor
    eps: float

    def __call__(self, x):
        """Normalizes `x` over its last dim and scales it."""
        return self.weight * (x / F.sqrt(F.mean(x * x, -1, keepdim=True) + self.eps))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_float_fields_are_runtime_scalars(device, triton_on_cuda):
    """A module's float field is a runtime scalar: another value runs the same
    plan (no trace, no kernel compiled), read by the kernel, not baked in; an
    RMS norm with it is still one kernel on MPS."""
    x, w = rand(8, 300) * 0.3, rand(300, seed=1)
    traces = []

    def forward(m, a):
        """Counts the trace and applies `m`."""
        traces.append(1)
        return m(a)

    weight = lumen.empty([300], device="meta")
    f = lumen.compile(forward, device=device)
    try:
        f(ScaledNorm(weight, 1e-6), lumen.empty([8, 300], device="meta"))
    except RuntimeError as e:
        pytest.skip(str(e))
    ScaledNorm(weight, 1e-6).load_state_dict({"weight": lumen.from_numpy(w)})
    for eps in (1e-6, 0.1, 1.0):
        out = lumen.to_numpy(f(ScaledNorm(weight, eps), lumen.from_numpy(x)))
        expected = w * x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    assert len(traces) == 1
    graph = lumen.make_graph(forward)(ScaledNorm(weight, 0.5), lumen.from_numpy(x))
    plan = lumen.graph_compiler.Plan(graph, device, parameters=[2], scalars=[1])
    workspace = lumen.Tensor.empty([plan.workspace_bytes], "uint8", device)
    inputs = [lumen.from_numpy(x).to(device), None, lumen.from_numpy(w).to(device)]
    for eps in (1e-6, 0.1, 1.0):
        inputs[1] = lumen.Tensor.full([], eps, "float32")
        (out,) = plan.run_in(workspace, inputs)
        expected = w * x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps)
        np.testing.assert_allclose(lumen.to_numpy(out), expected, rtol=1e-5, atol=1e-6)
    (step,) = [s for s in plan.steps() if ("s1", "float32", []) in s["inputs"]]
    if device == "mps":
        assert len(plan.steps()) == 1
        assert "constant float &in1" in step["fusion"]["source"]


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_programs_run_in_their_dtypes(device, triton_on_cuda):
    """Every op computes in its traced dtype: bfloat16 math (exp, log, sqrt,
    tanh, logistic, softmax) has no MPS or Triton kernel and raises there (the CPU's
    reference rounds it once); a dot accumulates in
    its required accum_dtype (float32 for 16-bit floats, written in the
    program; `@` always float32; sum and mean in their input's dtype). A
    dot returns its required output_dtype (`@`: its inputs'), a sum its
    accum_dtype."""
    try:
        x = lumen.from_numpy(rand(4, 8)).to(device).to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    for f in (F.exp, F.sqrt, lambda a: F.softmax(a, -1), F.tanh):
        if device not in ("mps", "cuda"):
            assert lumen.compile(f)(x).dtype == "bfloat16"
            continue
        with pytest.raises(RuntimeError, match="convert to float32"):
            lumen.compile(f)(x)
    out = lumen.compile(lambda a: F.exp(a.float()))(x)
    assert out.dtype == "float32"
    with pytest.raises(TypeError):
        prims.dot_general(x, x, (((1,), (1,)), ((), ())))
    w = lumen.from_numpy(rand(8, 3, seed=1)).to(device).to(dtype="bfloat16")
    dot = lambda a, b, d, o: prims.dot_general(a, b, (((1,), (0,)), ((), ())), d, o)  # noqa: E731
    a, b = (lumen.to_numpy(t.to(dtype="float32")).astype(np.float64) for t in (x, w))
    wide = lumen.compile(lambda a, b: dot(a, b, "float32", "float32"))(x, w)
    assert wide.dtype == "float32"
    np.testing.assert_allclose(lumen.to_numpy(wide), a @ b, rtol=1e-6, atol=1e-6)
    narrow = lumen.compile(lambda a, b: dot(a, b, "float32", "bfloat16"))(x, w)
    assert narrow.dtype == "bfloat16"
    np.testing.assert_allclose(lumen.to_numpy(narrow.to(dtype="float32")), a @ b, rtol=2**-8, atol=2**-8)
    if device == "cuda":
        with pytest.raises(Exception, match="accumulates in float32 or float16"):
            lumen.compile(lambda a, b: dot(a, b, "bfloat16", "bfloat16"))(x, w)
    else:
        assert lumen.compile(lambda a, b: dot(a, b, "bfloat16", "bfloat16"))(x, w).dtype == "bfloat16"
    with pytest.raises(ValueError, match="output_dtype"):
        lumen.compile(lambda a, b: dot(a, b, "bfloat16", "float32"))(x, w)
    graph = lumen.make_graph(lambda a, b: F.matmul(a, b, "float32", a.dtype))(x, w)
    assert "accum_dtype=f32 output_dtype=bf16" in str(graph)
    assert lumen.compile(lambda a, b: F.matmul(a, b, "float32", a.dtype))(x, w).dtype == "bfloat16"
    graph = lumen.make_graph(lambda a: F.sum(a, -1))(x)
    assert [n["primitive"] for n in graph.nodes()] == ["reduce_sum"]
    assert "accum_dtype=bf16" in str(graph)
    assert lumen.compile(lambda a: F.sum(a, -1))(x).dtype == "bfloat16"
    assert lumen.compile(lambda a: F.mean(a))(x).dtype == "bfloat16"
    total = lumen.compile(lambda a: F.sum(a.float(), -1))(x)
    assert total.dtype == "float32"
    np.testing.assert_allclose(lumen.to_numpy(total), a.sum(-1), rtol=1e-6, atol=1e-6)
    back = lumen.compile(lambda a: F.sum(a.float(), -1).bfloat16())(x)
    assert back.dtype == "bfloat16"
    np.testing.assert_allclose(lumen.to_numpy(back.to(dtype="float32")), a.sum(-1), rtol=2**-8, atol=2**-8)


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
    """On MPS an RMS norm in float32 of a bfloat16 input, cast back and
    scaled by its weight, is one row kernel: the cast and the weight run in
    its last pass (an epilogue). Rows of up to 8 elements a thread read x
    once, keeping x.float() (its traced dtype) in registers for the last
    pass. It agrees with the CPU to a bfloat16 rounding."""
    try:
        lumen.zeros([1], device=device)
    except RuntimeError as e:
        pytest.skip(str(e))
    weight = lumen.empty([n], dtype="bfloat16", device="meta")
    if device == "mps":
        graph = lumen.make_graph(lambda m, a: m(a))(
            UpcastNorm(weight, 1e-6), lumen.empty([8, n], dtype="bfloat16", device="meta")
        )
        (step,) = lumen.graph_compiler.Plan(graph, "mps", parameters=[2], scalars=[1]).steps()
        source = step["fusion"]["source"]
        assert step["label"].endswith("div → cast(float32 -> bfloat16) → broadcast_in_dim → mul"), step["label"]
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


@pytest.mark.mps
def test_split_reduction_converts_its_input_once():
    """``F.sum(x.float(), -1)`` of bfloat16 rows too long for one launch is two:
    the fused kernel converts each element once and writes float32
    partials; the second, the plain float32 kernel, sums those partials,
    converting nothing (a second convert of float32 values would change
    no value, so it is checked by what each launch runs)."""
    try:
        x = lumen.ones([4, 200_000], device="mps").to(dtype="bfloat16")
    except RuntimeError as e:
        pytest.skip(str(e))
    f = lumen.compile(lambda a: F.sum(a.float(), -1))
    graph = lumen.make_graph(lambda a: F.sum(a.float(), -1))(x)
    (step,) = lumen.graph_compiler.Plan(graph, "mps").steps()
    assert step["label"] == "cast(bfloat16 -> float32) → reduce_sum" and step["scratch"] is not None
    source = step["fusion"]["source"]
    assert source.count("convert_value<float>") == 1, source
    assert "device float *out" in source and "device const bfloat *in0" in source, source
    f(x)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS], record_shapes=True) as prof:
        out = f(x)
        lumen.mps.synchronize()
    first, last = [e for e in prof.events() if e["kind"] == "gpu" and not e["name"].startswith("copy")]
    assert first["kernel"] == step["fusion"]["kernel"]
    assert last["kernel"] == "reduce_sum_rows_f32"
    assert last["inputs"] == first["outputs"] and last["inputs"][0][0] == "float32"
    np.testing.assert_array_equal(lumen.to_numpy(out), np.full(4, 200_000, np.float32))


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
    written: one row kernel on MPS, its reductions inside, agreeing with the
    CPU. Nothing matches any one normalization. With no backward to read
    its values of a row (an RMS norm's sqrt(mean + eps)), it writes its
    output alone."""
    x, w = rand(64, 300), rand(300, seed=1)
    try:
        X, W = lumen.from_numpy(x).to(device), lumen.from_numpy(w).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        (step,) = lumen.graph_compiler.Plan(lumen.make_graph(f)(X, W), "mps").steps()
        assert step["fusion"] is not None and "reduce_" in step["label"], step["label"]
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
    if device == "mps":
        labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()]
        if "sqrt" in str(labels):
            assert sum("sqrt" in label for label in labels) == 1, labels
    got = lumen.compile(grad)(X, W, C)
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-5)


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
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(loss)(H, W, T), "mps").steps()
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
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(loss)(H, W, T), "mps").steps()
        assert "linear_cross_entropy" in [s["label"] for s in steps], [s["label"] for s in steps]
        assert all(s["output"][2] != [n, vocab] for s in steps), [s["label"] for s in steps]
        steps = lumen.graph_compiler.Plan(lumen.make_graph(both)(H, W, T), "mps").steps()
        assert "linear_cross_entropy" not in [s["label"] for s in steps], [s["label"] for s in steps]
    hr = lumen.to_numpy(H.to(dtype="float32")).astype(np.float64)
    wr = lumen.to_numpy(W.to(dtype="bfloat16").to(dtype="float32")).astype(np.float64)
    x = hr @ wr.T
    m = x.max(-1, keepdims=True)
    want = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0] - x[np.arange(n), t]
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(loss)(H, W, T)), want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lumen.to_numpy(lumen.compile(both)(H, W, T)[0]), want, rtol=1e-5, atol=1e-5)


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
def test_chunked_linear_cross_entropy(n, d, vocab, chunk, reduce, head):
    """A cross entropy of a matmul's logits trained with
    ``lumen.config.compiler.mps.fused_linear_cross_entropy_chunk_size``
    (``lumen/autograd/cross_entropy.py``): its backward a chunk of rows at
    a time (4 of 16 rows of 64; 8 of 128 rows of 1000, the last one
    short; 32 of 32 rows of 1024, more than a kernel binds, its losses
    joined a few chunks' at a time; or one chunk, all 64 rows: its logits
    the forward's, the program unchunked). Its weight bfloat16, or float32
    cast to bfloat16 in the step (an LM head's master weight, its gradient
    float32). On MPS each chunk's logits computed once (XMA's fused linear
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
        H, W, T, C = (lumen.from_numpy(a).to("mps") for a in (h, w, t, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    H, W = H.to(dtype="bfloat16"), W.to(dtype=head)
    step = trained(unfused)
    lumen.config.compiler.mps.fused_linear_cross_entropy_chunk_size = chunk
    try:
        plan = lumen.graph_compiler.Plan(lumen.make_graph(step)(H, W, T, C), "mps")
        got = [lumen.to_numpy(v.to(dtype="float32")) for v in lumen.compile(step)(H, W, T, C)]
    finally:
        lumen.config.compiler.mps.fused_linear_cross_entropy_chunk_size = None
    sizes = [min(chunk, n - start) for start in range(0, n, chunk)]
    steps = plan.steps()
    labels = [s["label"] for s in steps]
    baseline = lumen.graph_compiler.Plan(lumen.make_graph(trained(unfused))(H, W, T, C), "mps")
    if len(sizes) == 1:
        chunked = [label for label in labels if label not in ("full", "dynamic_update_slice")]
        assert sorted(chunked) == sorted(s["label"] for s in baseline.steps()), labels
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
        one = vocab <= 64
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
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(step)(X, W, C), "mps").steps()
        (forward,) = [s for s in steps if "sqrt" in s["label"]]
        assert [o[2] for o in forward["extra_outputs"]] == [[64, 1]], forward["label"]
    got = lumen.compile(step)(X, W, C)
    want = lumen.compile(step, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, e in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


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
    if device == "mps" and kernels is not None:
        steps = lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
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
    them as it goes (a threadgroup a block of rows, each column's sum in
    registers), writing a block's sums each, then one kernel sums those
    (a block at a time, in order): two kernels for the backward, the sums
    the same bits every run, the gradients the CPU's."""
    x, w, c = rand(1024, 64), rand(64, seed=1), rand(1024, 64, seed=2)
    try:
        X, W, C = (lumen.from_numpy(a).to(device) for a in (x, w, c))
    except RuntimeError as e:
        pytest.skip(str(e))
    grad = lumen.grad(f, (0, 1))
    if device == "mps":
        steps = lumen.graph_compiler.Plan(lumen.make_graph(grad)(X, W, C), "mps").steps()
        backward = [s for s in steps if s["extra_outputs"] and s["extra_outputs"][0][2] == [64, 64]]
        assert len(backward) == 1, [s["label"] for s in steps]
        assert len(backward[0]["extra_outputs"]) == sums
        final = [s for s in steps if s["primitive"] == "reduce_sum"]
        assert len(final) == sums and all(s["inputs"][0][2] == [64, 64] for s in final)
    got = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    again = [lumen.to_numpy(g) for g in lumen.compile(grad)(X, W, C)]
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in (x, w, c)))
    for g, a, e in zip(got, again, want):
        np.testing.assert_array_equal(g, a)
        np.testing.assert_allclose(g, lumen.to_numpy(e), rtol=1e-4, atol=1e-4)


def test_ops_are_functions():
    """The ops are functions in lumen.functional (``import lumen.functional
    as F``), as torch's are; a traced tensor keeps only its operators, layout
    and dtype casts, and lumen itself re-exports no op."""
    ops = ["exp", "log", "sqrt", "tanh", "sigmoid", "relu", "sum", "mean", "amax", "max", "softmax"]
    ops += ["log_softmax", "rms_norm", "matmul", "where", "maximum", "minimum", "add", "mul", "div", "eq", "gt"]
    for name in ops:
        assert callable(getattr(F, name)) and name in F.__all__
        assert not hasattr(lumen.graph.TracedTensor, name), name
        assert not hasattr(lumen, name), name
    for name in ["reshape", "permute", "transpose", "t", "unsqueeze", "flatten", "split", "to", "float", "__add__"]:
        assert hasattr(lumen.graph.TracedTensor, name), name
    assert lumen.functional is F
    a = lumen.zeros([2, 3])
    by_operator = lumen.make_graph(lambda x: F.matmul((x + 1) * x, x.t(), "float32", x.dtype))(a)
    by_function = lumen.make_graph(lambda x: F.matmul(F.mul(F.add(x, 1), x), x.t(), "float32", "float32"))(a)
    assert str(by_operator) == str(by_function)


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
    ],
    ids=["bias relu", "batched residual", "folded residual", "relu cast", "times exp(y)", "silu in float32"],
)
def test_matmul_epilogues_fuse(device, f, shapes, dtype, triton_on_cuda):
    """The elementwise primitives after a matmul (a bias, a residual, an
    activation, a cast, a product with another fused value) run in its
    kernel as it writes each output (``contraction_epilogues``): one step,
    exactly the unfused plan's result. On CUDA (Triton), the CPU's within
    rounding; float16 math has no Triton kernel."""
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
    read again; exactly the unfused plan's result. Wider rows: the matmul's
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


@pytest.mark.mps
@pytest.mark.parametrize("m, tiles", [(64, "mid"), (192, "mid"), (16, "small"), (256, "large")])
def test_matmul_tiles_fit_its_rows(m, tiles):
    """Untuned (``autotune`` off), a float matmul runs on 128x64 tiles; on
    64x64 ones where M leaves 33 to 64 rows of a last one (M of 64, 192:
    half a large tile idle, the 32x32 tiles reading the operands twice as
    often); on 32x32 where it leaves 32 or fewer (or the large tiles are too
    few to fill the GPU). With an epilogue too (its kernel's ``_mid``,
    ``_small`` variant). Tuned, it runs on whichever tile was fastest. The
    results NumPy's either way."""
    rng = np.random.default_rng(0)
    k, n = 256, 2048
    try:
        a, b = (
            lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to("mps").to(dtype="bfloat16")
            for s in ((m, k), (k, n))
        )
    except RuntimeError as e:
        pytest.skip(str(e))
    product = lumen.to_numpy(a.to(dtype="float32")).astype(np.float64) @ lumen.to_numpy(b.to(dtype="float32"))

    def plain(a, b):
        """float32 matmul."""
        return F.matmul(a, b, "float32", "float32")

    def fused(a, b):
        """relu of the float32 matmul."""
        return F.relu(plain(a, b))

    for tuned in (False, True):
        lumen.config.compiler.mps.autotune = tuned
        try:
            for f, want in ((plain, product), (fused, np.maximum(product, 0))):
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
                    got = lumen.to_numpy(lumen.compile(f)(a, b))
                (kernel,) = {e["kernel"] for e in prof.events() if e["kind"] == "gpu"}
                ran = "small" if "small" in kernel else "mid" if "mid" in kernel else "large"
                assert tuned or ran == tiles, kernel
                np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)
        finally:
            lumen.config.compiler.reset()


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("lhs_k, rhs_k", [(1, 0), (1, 1), (0, 0), (0, 1)])
@pytest.mark.parametrize("m, k, n", [(512, 256, 512), (300, 200, 260), (5, 4096, 1000), (1, 4096, 4096)])
def test_tuned_matmul_matches_numpy_in_every_layout(device, dtype, lhs_k, rhs_k, m, k, n, triton_on_cuda):
    """Tuned, a float matmul of 64K or more outputs, or of a few rows,
    runs on whichever candidate was fastest: lumen's simdgroup kernels,
    where Metal Performance Primitives compile ``matmul_mpp``, and for a
    few rows ``matmul_skinny``. Either operand may be transposed
    (contracting its first dimension rather than its last, or its last
    rather than its first), tiles may be left partial at the edges (300 x
    260 of 64-wide tiles, 5 rows of 8-row ones), and the result may round
    to bfloat16. Each matches NumPy, on CUDA (Triton's GEMM) too."""
    rng = np.random.default_rng(0)
    shapes = ((m, k) if lhs_k == 1 else (k, m), (k, n) if rhs_k == 0 else (n, k))
    try:
        a, b = (lumen.from_numpy(rng.standard_normal(s).astype(np.float32)).to(device).to(dtype=dtype) for s in shapes)
    except RuntimeError as e:
        pytest.skip(str(e))
    x, y = (lumen.to_numpy(t.to(dtype="float32")).astype(np.float64) for t in (a, b))
    want = (x if lhs_k == 1 else x.T) @ (y if rhs_k == 0 else y.T)
    dims = (((lhs_k,), (rhs_k,)), ((), ()))
    got = lumen.compile(lambda a, b: prims.dot_general(a, b, dims, "float32", dtype))(a, b)
    rtol = 1e-4 if dtype == "float32" else 2**-6
    np.testing.assert_allclose(lumen.to_numpy(got.to(dtype="float32")), want, rtol=rtol, atol=rtol * 16)


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


def test_division_by_a_scalar_is_a_product_with_its_reciprocal():
    """``x / c`` (a Python number, or a runtime scalar) traces as
    ``x * (1 / c)``; by zero, and of tensors, it stays a division."""
    x = lumen.from_numpy(np.random.default_rng(0).standard_normal(1000).astype(np.float32))
    graph = str(lumen.make_graph(lambda x: x / 3)(x))
    assert "mul" in graph and "div" not in graph, graph
    out = lumen.to_numpy(lumen.compile(lambda x: x / 3)(x))
    np.testing.assert_array_equal(out, lumen.to_numpy(x) * np.float32(1 / 3))
    graph = str(lumen.make_graph(lambda x, s: x / s)(x, 3.0))
    assert graph.count("div") == 1 and "mul" in graph, graph
    assert "div" in str(lumen.make_graph(lambda x: x / 0.0)(x))
    assert "div" in str(lumen.make_graph(lambda x: x / x)(x))


def test_matmul_takes_accum_and_output_dtypes():
    """F.matmul takes ``accum_dtype`` and ``output_dtype`` (required); ``@``
    infers them (float32 accumulation, the inputs' dtype)."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")
    y = lumen.empty([8, 3], dtype="bfloat16", device="meta")
    assert "accum_dtype=f32 output_dtype=bf16" in str(
        lumen.make_graph(lambda x, y: F.matmul(x, y, "float32", x.dtype))(x, y)
    )
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "float32", "float32"))(x, y)
    assert "accum_dtype=f32 output_dtype=f32" in str(g)
    g = lumen.make_graph(lambda x, y: F.matmul(x, y, "bfloat16", "bfloat16"))(x, y)
    assert "accum_dtype=bf16 output_dtype=bf16" in str(g)
    with pytest.raises(TypeError, match="output_dtype"):
        lumen.make_graph(lambda x, y: F.matmul(x, y))(x, y)
    a, b = (lumen.ones(s, dtype="bfloat16") for s in ([4, 8], [8, 3]))
    assert lumen.compile(lambda x, y: F.matmul(x, y, "float32", "float32"))(a, b).dtype == "float32"


def test_rounding_a_contraction_then_widening_it_warns():
    """A matmul or attention's scores output in bf16 (accumulated in
    float32) and cast back to float32, through a scale, warns: outputting
    float32 is more accurate. Outputting float32, or not widening, does
    not."""
    x = lumen.empty([4, 8], dtype="bfloat16", device="meta")

    def scores(x):
        s = F.matmul(x, x.t(), "float32", x.dtype) * 0.5
        return s.float()

    with pytest.warns(UserWarning, match=rf"cast at {re.escape(__file__)}:\d+") as record:
        lumen.make_graph(scores)(x)
    (w,) = record
    assert (w.filename, w.lineno) == (__file__, scores.__code__.co_firstlineno + 1)
    with pytest.warns(
        UserWarning,
        match=r"dot_general\(bf16\[4,8\], bf16\[8,4\]\) -> bf16\[4,4\] accumulates in f32 but outputs bf16, then dot_general → mul → cast\(bfloat16 -> float32\)",
    ):
        lumen.make_graph(lambda x: (F.matmul(x, x.t(), "float32", x.dtype) * 0.5).float())(x)
    with pytest.warns(UserWarning, match="output_dtype=f32"):
        lumen.compile(lambda x: F.softmax(F.matmul(x, x.t(), "float32", x.dtype).float(), -1))(x)
    q = lumen.empty([1, 4, 2, 8], dtype="bfloat16", device="meta")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lumen.make_graph(lambda x: F.matmul(x, x.t(), "float32", "float32") * 0.5)(x)
        lumen.make_graph(lambda x: F.relu(F.matmul(x, x.t(), "float32", x.dtype)))(x)
        lumen.make_graph(lambda q: F.flash_attention(q, q, q))(q)


class _Counter(lumen.nn.Module):
    """A step counter and a weight the step updates."""

    n: lumen.Tensor
    w: lumen.Tensor


def test_copy_into_a_weight_is_written_back():
    """w.copy_(value) in a compiled function assigns a module's weight: the
    compiled function writes it back after each call (an optimizer's step),
    and the next call reads it; a call compiling on meta tensors does not."""
    counter = _Counter(lumen.empty([], device="meta"), lumen.empty([3], device="meta"))

    def step(c, x):
        """Increments the counter and adds `x` times it to the weight."""
        c.n.copy_(c.n + 1.0)
        c.w.copy_(c.w + x * c.n)
        return c.n * 1.0

    step = lumen.compile(step, device="cpu")
    step(counter, lumen.empty([3], device="meta"))
    x = np.array([1.0, 2.0, 3.0], np.float32)
    for k in range(1, 4):
        assert float(lumen.to_numpy(step(counter, lumen.from_numpy(x)))) == k
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda c: c.w * 1.0)(counter)), 6 * x)
    with pytest.raises(TypeError, match="copy_"):
        lumen.make_graph(lambda c: c.w.copy_(c.n))(counter)


class _Steps(lumen.nn.Module):
    """A weight, a float step count and a tuple of learning rates."""

    w: lumen.Tensor
    t: float
    lrs: tuple


def test_copy_into_a_modules_float_is_written_back():
    """A module's float (a runtime scalar) assigned with copy_ is written
    back into the module, on the host, after each call: the next call
    passes it (a step count). One in a tuple cannot be."""
    steps = _Steps(lumen.empty([2], device="meta"), 0.0, (0.5,))

    def step(s):
        """Increments the step and adds it to the weight."""
        s.t.copy_(s.t + 1.0)
        s.w.copy_(s.w + s.t)
        return s.t * 1.0

    step = lumen.compile(step, device="cpu")
    for k in range(1, 4):
        step(steps)
        assert steps.t == k
    np.testing.assert_array_equal(lumen.to_numpy(lumen.compile(lambda s: s.w * 1.0)(steps)), [6.0, 6.0])

    def assign_tuple(s):
        """Assigns a float inside a tuple field, which raises."""
        s.lrs[0].copy_(s.lrs[0] * 2.0)
        return s.w * 1.0

    with pytest.raises(TypeError, match="tuple"):
        lumen.compile(assign_tuple, device="cpu")(steps)


class _Scalars(lumen.nn.Module):
    """A weight with float, int and bool fields."""

    w: lumen.Tensor
    lr: float
    step: int
    clip: bool
    heads: int


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_scalar_fields_are_assigned_as_python_values(device, triton_on_cuda):
    """A module's float, int and bool fields are assigned as Python values,
    inside a compiled function (``s.step = s.step + 1``) and outside, and
    written back as their type. A float is a runtime scalar; an int or bool
    is static until the function assigns it or it changes, then a runtime
    scalar (traced again once: no plan per value); one that cannot be (a
    condition, a shape) stays static, a plan per value."""
    traces = []

    def step(s, x):
        """Counts the trace, updates the step and learning rate, and uses every field."""
        traces.append(1)
        s.step = s.step + 1
        s.lr = s.lr * 0.5
        y = x * s.lr + s.step
        if s.clip:
            y = F.maximum(y, 0.0)
        return y.reshape([s.heads, -1])

    f = lumen.compile(step, device=device)
    s = _Scalars(lumen.empty([4], device="meta"), 1.0, 0, False, 2)
    try:
        x = lumen.from_numpy(np.full(4, -3.0, np.float32)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    for k in range(1, 5):
        y = lumen.to_numpy(f(s, x))
        np.testing.assert_allclose(y, np.full((2, 2), -3.0 * 0.5**k + k), rtol=1e-6)
        assert (s.step, s.lr, s.clip) == (k, 0.5**k, False)
        assert (type(s.step), type(s.lr), type(s.clip)) == (int, float, bool)
    assert len(traces) == 2
    s.lr, s.clip = 1.0, True
    np.testing.assert_allclose(lumen.to_numpy(f(s, x)), np.full((2, 2), 3.5), rtol=1e-6)
    s.clip = False
    f(s, x)
    s.clip = True
    f(s, x)
    assert len(traces) == 4
    s.heads = 4
    assert list(f(s, x).shape) == [4, 1]
    with pytest.raises(TypeError, match="int field"):
        s.step = 1.5


class _Seen(lumen.nn.Module):
    """A weight, a float set from the device and a step counter."""

    w: lumen.Tensor
    seen: float
    step: int


def test_scalars_written_from_scalars_alone_run_on_the_host():
    """A scalar field assigned from scalars and constants alone (``step +
    1``) is computed on the host each call, from the scalars passed, never
    read back from the device (which would wait for the device's work); one
    assigned from the device's values (a loss) is read back."""
    from lumen.graph import tracer

    def step(s, x):
        """Increments the step on the host and sets `seen` from the device."""
        s.step = s.step + 1
        s.seen = F.sum(x * s.w)
        return x * 1.0

    s = _Seen(lumen.empty([3], device="meta"), 0.0, 1)
    runtime = tracer._runtime({(0, ("step",))})
    args = (s, lumen.empty([3], device="meta"))
    _, _, (_, read, _, (_, _, host)), _, _, _ = tracer._trace(step, args, "cpu", runtime, split=True)
    assert (read, host) == ([(0, ("seen",))], [(0, ("step",))])
    f = lumen.compile(step, device="cpu")
    f(s, lumen.empty([3], device="meta"))
    s.load_state_dict({"w": lumen.from_numpy(np.full(3, 2.0, np.float32))})
    for _ in range(2):
        f(s, lumen.from_numpy(np.arange(3, dtype=np.float32)))
    assert (s.seen, s.step) == (6.0, 3) and type(s.step) is int


def test_float64_runtime_scalars_are_exact():
    """A float runtime scalar of a float64 program on the CPU is the Python
    float exactly, not rounded to float32 first; float32 programs see it
    rounded once."""
    f = lumen.compile(lambda x, c: x * c, device="cpu")
    assert lumen.to_numpy(f(lumen.from_numpy(np.ones(1)), 0.1))[0] == 0.1
    assert lumen.to_numpy(f(lumen.from_numpy(np.ones(1, np.float32)), 0.1))[0] == np.float32(0.1)


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


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_concatenate_runs_unfused(device, triton_on_cuda):
    """With fusion off, a concatenate (alone a fusion on MPS) still runs."""
    try:
        a = lumen.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3)).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    graph = lumen.make_graph(lambda a: prims.concatenate([a, a * 2.0], 1))(a)
    plan = lumen.graph_compiler.Plan(graph, device, fuse=False)
    (out,) = plan.run([a])
    np.testing.assert_array_equal(lumen.to_numpy(out), np.concatenate([lumen.to_numpy(a), 2 * lumen.to_numpy(a)], 1))


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_cpu_inside_a_compiled_function(device, triton_on_cuda):
    """``x.cpu()`` (and ``x.to("cpu")``, with a dtype too) inside a compiled
    function, as in PyTorch: a traced tensor has a device (the plan's, or
    ``cpu`` once on the host), and ``cpu()`` of a CPU tensor is the tensor
    itself (assigning it assigns the argument); another device's is copied
    to the host, its own tensor, returned as a CPU tensor."""
    devices = []

    def f(x):
        """Moves values to the host inside the trace and records each one's device."""
        y = x * 2.0
        h = y.cpu()
        devices.extend([x.device, h.device, (h + 1.0).device, h.to(device).device])
        return h, F.sum(h + 1.0).to("cpu"), x.to("cpu", "float16"), x.to(device="cpu"), y, x.cpu().add_(0.5), x

    try:
        g = lumen.compile(f, device=device)
        first = g(lumen.tensor([1.0, 2.0]))
    except RuntimeError as e:
        pytest.skip(str(e))
    assert devices == [f"{device}:0", "cpu:0", "cpu:0", f"{device}:0"]
    h, total, half, moved, y, added, x = first
    assert all(t.device == "cpu:0" for t in (h, total, half, moved, added))
    assert y.device == x.device == f"{device}:0"
    assert half.dtype == "float16"
    assert [h.tolist(), total.tolist(), half.tolist()] == [[2.0, 4.0], [8.0], [1.0, 2.0]]
    expected = [1.5, 2.5] if device == "cpu" else [1.0, 2.0]
    assert added.tolist() == [1.5, 2.5] and moved.tolist() == x.tolist() == expected


def test_python_values_inside_a_compiled_function_raise():
    """A traced tensor has no value while tracing: ``item``, ``tolist`` and
    ``float`` raise (``cpu`` keeps a tensor: it works)."""
    for body in (lambda x: x.item(), lambda x: x.tolist(), lambda x: float(F.sum(x))):
        with pytest.raises((AttributeError, TypeError)):
            lumen.compile(body)(lumen.tensor([1.0]))


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_cpu_inside_a_compiled_function_transfers(device, triton_on_cuda):
    """``.cpu()`` inside a compiled MPS function splits it into stages: the
    ops before it on the device, then its copy to the host (``to_host``:
    ``lumen::copy_d2h``, its ``lumen::wait`` inside), then the ops reading
    it on the host, compiled for the CPU, as PyTorch runs ops on CPU
    tensors (no kernels), the result a CPU tensor. An op reading host and
    device values raises PyTorch's error. The gradient comes back to the
    device (``to_device``)."""

    def mid(x):
        """Device work, a copy to the host, then host work."""
        return (x * 3.0).cpu() * 2.0 + 1.0

    x = lumen.tensor([1.0, 2.0])
    try:
        f = lumen.compile(mid, device=device)
        f(x)
    except RuntimeError as e:
        pytest.skip(str(e))
    if device == "mps":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MPS]) as prof:
            y = f(x)
        events = prof.events()
        by_id = {e["id"]: e for e in events}
        assert [e["name"] for e in events if e["kind"] == "gpu"] == ["copy_h2d", "mul"]
        (copy,) = [e for e in events if e["name"] == "lumen::copy_d2h"]
        names = []
        while copy.get("parent") in by_id:
            copy = by_id[copy["parent"]]
            names.append(copy["name"])
        assert "to_host" in names
    else:
        y = f(x)
    assert y.device == "cpu:0" and y.tolist() == [7.0, 13.0]
    with pytest.raises(RuntimeError, match="Expected all tensors to be on the same device"):
        lumen.compile(lambda x: x.cpu() + x, device=device)(x)
    grad = lumen.compile(lumen.grad(lambda x: F.sum((x * 3.0).cpu() * 5.0).to(device)), device=device)(x)
    assert grad.device == f"{device}:0" and grad.tolist() == [15.0, 15.0]
