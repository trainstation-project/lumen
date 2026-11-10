"""Tests for gather, scatter_add and take_along_axis: values, fusion, gradients, determinism and operand checks."""

import pathlib
import tempfile

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU

import lumen
import lumen.functional as F
from lumen import prims

DEVICES = ["cpu", MPS, CUDA, TPU]


@pytest.fixture(autouse=True)
def triton_backend():
    """Each test with CUDA's programs on the Triton backend."""
    with lumen.config.compiler.patch(use_triton_backend=True):
        yield


def _tensor(a, device, dtype=None):
    """Returns numpy array ``a`` on ``device`` (cast to ``dtype`` when given), skipping the test when the device is unavailable."""
    try:
        t = lumen.from_numpy(a).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    return t if dtype is None else t.to(dtype=dtype)


# Indices in range of every axis gathered along (3 to 5 long), repeating some and leaving the last rows unread.
IDS = np.array([[2, 0, 2], [1, 0, 1]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
@pytest.mark.parametrize("axis", [0, 1])
def test_gather(device, index_dtype, dtype, axis):
    """The entries the indices pick along the axis, the
    axis replaced by the indices' shape, for any dtype; ``weight[ids]`` is
    one along the first."""
    x = (np.random.default_rng(0).standard_normal((4, 5, 3)) * 4).astype(np.float32)
    xt = _tensor(x, device, dtype)
    x = lumen.to_numpy(xt.to(dtype="float32"))
    ids = _tensor(IDS.astype(index_dtype), device)
    got = lumen.compile(lambda x, i: prims.gather(x, i, axis))(xt, ids)
    want = np.take(x, IDS, axis=axis)
    np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)
    if axis == 0:
        got = lumen.compile(lambda x, i: x[i])(xt, ids)
        np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gather_fuses(device, index_dtype, axis):
    """A gather computes its elements in the kernel reading them (an elementwise
    fusion's, or a row kernel's: an embedding's lookup, its positions added,
    then normalized), reading the operand at each picked index:
    no kernel of its own; the values the CPU's."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5, 3)).astype(np.float32)
    xt, ids = _tensor(x, device), _tensor(IDS.astype(index_dtype), device)

    def f(x, i):
        """Returns a gather of ``x * 2`` plus one, and that result RMS-normalized."""
        g = prims.gather(x * 2.0, i, axis) + 1.0
        return g, F.rms_norm(g, g.shape[-1])

    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(xt, ids), device).steps()]
    assert "gather" not in labels and any("gather" in label for label in labels), labels
    got = lumen.compile(f)(xt, ids)
    want = lumen.compile(f, device="cpu")(lumen.from_numpy(x), lumen.from_numpy(IDS.astype(index_dtype)))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), lumen.to_numpy(w), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", [MPS, CUDA])
def test_gather_of_a_table_of_as_many_elements_as_rows(device):
    """A lookup in a row kernel (an embedding, its positions added, then
    normalized) whose table has as many elements as the kernel has rows
    (16 x 64 entries, 32 x 32 tokens): read at its ids, not as a value a
    row (a saved statistic is), its gradient the CPU's too."""
    rng = np.random.default_rng(5)
    table, positions = rng.standard_normal((16, 64)).astype(np.float32), rng.standard_normal((32, 64)).astype(
        np.float32
    )
    ids, c = rng.integers(0, 16, (32, 32)), rng.standard_normal((1024, 64)).astype(np.float32)

    def loss(table, positions, ids, c):
        """Returns the summed product of ``c`` with the RMS-normalized embedding of ``ids`` plus ``positions``."""
        h = (table[ids] + positions).reshape(1024, 64)
        return F.sum(F.rms_norm(h, 64) * c)

    grad = lumen.value_and_grad(loss, (0, 1))
    args = (table, positions, ids, c)
    got = lumen.compile(grad)(*(_tensor(a, device) for a in args))
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in args))
    np.testing.assert_allclose(got[0].item(), want[0].item(), rtol=1e-5)
    for g, w in zip(got[1], want[1]):
        np.testing.assert_allclose(lumen.to_numpy(g.to("cpu")), lumen.to_numpy(w), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("axis", [0, 1])
def test_scatter_add(device, dtype, axis):
    """Each update added at its index, those at
    one entry summed."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 5, 3)) * 4).astype(np.float32)
    u = (rng.standard_normal(np.take(x, IDS, axis=axis).shape) * 4).astype(np.float32)
    xt, ut = _tensor(x, device, dtype), _tensor(u, device, dtype)
    x, u = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, ut))
    ids = _tensor(IDS.astype(np.int64), device)
    got = lumen.compile(lambda x, i, u: prims.scatter_add(x, i, u, axis))(xt, ids, ut)
    want = x.copy()
    np.add.at(want, (slice(None),) * axis + (IDS,), u)
    np.testing.assert_allclose(lumen.to_numpy(got.to(dtype="float32")), want, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gather_gradient(device, dtype):
    """A lookup's gradient: each row's cotangent summed over its reads,
    zeros where none read it, accumulated in float32."""
    rng = np.random.default_rng(2)
    w, c = rng.standard_normal((4, 3)).astype(np.float32), rng.standard_normal((2, 3, 3)).astype(np.float32)
    wt, ct = _tensor(w, device, dtype), _tensor(c, device, dtype)
    w, c = (lumen.to_numpy(t.to(dtype="float32")) for t in (wt, ct))
    ids = _tensor(IDS.astype(np.int64), device)
    grad = lumen.grad(lambda w, i, c: F.sum((w[i] * c).float()))
    got = lumen.compile(grad)(wt, ids, ct)
    want = np.zeros_like(w)
    np.add.at(want, IDS, c)
    assert got.dtype == dtype
    tol = {"float32": 1e-6, "bfloat16": 1e-2}[dtype]
    np.testing.assert_allclose(lumen.to_numpy(got.to(dtype="float32")), want, rtol=tol, atol=tol)
    graph = str(lumen.make_graph(grad)(wt, ids, ct))
    assert "scatter_add[axis=0]" in graph and ("f32[4,3] = scatter_add" in graph), graph


def test_gather_checks_its_operands():
    """Indices are int32 or int64; the axis one of the operand's; a
    scatter's updates shaped as the gather at its indices."""
    m = lambda shape, dtype="float32": lumen.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, i: prims.gather(x, i, 1))(m([4, 6, 2]), m([3, 5], "int32"))
    assert "f32[4,3,5,2] = gather[axis=1]" in str(g)
    with pytest.raises(ValueError, match="int32 or int64"):
        lumen.make_graph(lambda x, i: prims.gather(x, i, 0))(m([4, 6]), m([3]))
    with pytest.raises(ValueError, match="axis 2"):
        lumen.make_graph(lambda x, i: prims.gather(x, i, 2))(m([4, 6]), m([3], "int64"))
    with pytest.raises(ValueError, match="updates"):
        lumen.make_graph(lambda x, i, u: prims.scatter_add(x, i, u, 0))(m([4, 6]), m([3], "int64"), m([3, 5]))


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize("width, rows, picks", [(64, 5, 300), (7, 5, 300), (7, 1, 2500)])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32", "int8"])
@lumen.config.compiler.patch(deterministic=True)
def test_scatter_add_is_deterministic(device, dtype, width, rows, picks):
    """A deterministic scatter_add (``deterministic``;
    by default it adds by atomics, as torch's ``index_add``) adds each element's
    updates in index order (MPS and Triton sort the picks by position, then a
    thread a column adds its row's in order; no atomics on the values): the
    CPU's result bit for bit, for any dtype (bfloat16 too, each sum rounded as
    the reference rounds it), the same each run; a lookup's gradient too
    (deterministic training). 2500 picks at one row are more than either sorts
    at once (1024), so that row walks every index in order instead."""
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((rows, width)) * 4).astype(np.float32)
    ids = rng.integers(0, rows, (picks,))
    u = (rng.standard_normal((picks, width)) * 4).astype(np.float32)
    f = lambda x, i, u: prims.scatter_add(x, i, u, 0)  # noqa: E731
    on = lambda device: [
        _tensor(a, device, dtype) if a.dtype == np.float32 else _tensor(a, device) for a in (x, ids, u)
    ]  # noqa: E731
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(*on("cpu")).to(dtype="float32"))
    compiled = lumen.compile(f, device=device)
    for _ in range(3):
        got = lumen.to_numpy(compiled(*on(device)).to(dtype="float32").to("cpu"))
        np.testing.assert_array_equal(got, want)


# Unit roundoff of each float dtype: the largest relative error of rounding one result.
ROUNDOFF = {"float64": 2.0**-53, "float32": 2.0**-24, "float16": 2.0**-11, "bfloat16": 2.0**-8}


@pytest.mark.parametrize("device", [MPS, CUDA])
@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "float16", "bfloat16", "int64", "uint64", "int32", "uint32", "int16", "int8"]
)
def test_scatter_add_atomic(device, dtype):
    """By default (not deterministic) a scatter_add adds each update element
    atomically where its dtype has atomic adds, as torch's ``index_add`` (MPS:
    float32, float16, bfloat16, int32 and uint32; Triton: those, float64, int64
    and uint64), the others as deterministic: with about 60 updates at each of
    5 rows, integers match the CPU bit for bit, and floats, summed in an
    arbitrary order, are within twice the worst-case rounding error of any
    order of their sum (``gamma(k) * sum|terms|`` for ``k`` adds) of the CPU's
    result. MPS has no float64."""
    if device == "mps" and dtype == "float64":
        pytest.skip("MPS has no float64")
    rng = np.random.default_rng(5)
    rows, width = 5, 67
    ids = rng.integers(0, rows, (300,))
    if dtype in ROUNDOFF:
        x, u = ((rng.standard_normal(shape) * 4).astype(np.float32) for shape in ((rows, width), (300, width)))
    else:
        low = -3 if dtype.startswith("int") else 0
        x, u = (rng.integers(low, 4, shape).astype(np.int32) for shape in ((rows, width), (300, width)))
    f = lambda x, i, u: prims.scatter_add(x, i, u, 0)  # noqa: E731
    on = lambda device: [_tensor(x, device, dtype), _tensor(ids, device), _tensor(u, device, dtype)]  # noqa: E731
    want = lumen.compile(f, device="cpu")(*on("cpu"))
    compiled = lumen.compile(f, device=device)
    args = on(device)
    got = compiled(*args).to("cpu")
    if dtype not in ROUNDOFF:
        np.testing.assert_array_equal(lumen.to_numpy(got), lumen.to_numpy(want))
    else:
        exact = lambda t: lumen.to_numpy(t.to("cpu").to(dtype="float64"))  # noqa: E731
        xs, us = exact(args[0]), exact(args[2])
        # Per element: the adds into it, and the sum of its terms' magnitudes.
        adds = np.bincount(ids, minlength=rows)[:, None]
        magnitude = np.abs(xs).copy()
        np.add.at(magnitude, ids, np.abs(us))
        k = adds * ROUNDOFF[dtype]
        bound = k / (1 - k) * magnitude
        np.testing.assert_array_less(np.abs(exact(got) - exact(want)), 2 * bound + 1e-300)
    if device == "cuda":
        with tempfile.TemporaryDirectory() as d:
            compiled.dump_code(d, *args)
            code = "".join(p.read_text() for p in pathlib.Path(d).rglob("*") if p.is_file())
        kernel = "scatter_add_rows_kernel" if dtype in ("int16", "int8") else "scatter_add_atomic_kernel"
        assert f"'s {kernel}," in code, code


def test_scatter_add_in_place():
    """Its value is its operand's memory where nothing reads the operand
    after it (a gradient's zeros): the updates added there alone."""
    args = [
        lumen.empty(s, dtype=d, device="meta") for s, d in (([4, 6], "float32"), ([3], "int64"), ([3, 6], "float32"))
    ]
    plan = str(
        lumen.graph_compiler.Plan(lumen.make_graph(lambda x, i, u: prims.scatter_add(x * 2.0, i, u, 0))(*args), "cpu")
    )
    assert "out0:f32[4,6] = scatter_add[axis=0] out0 " in plan, plan


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index_dtype", [np.int8, np.int32, np.int64])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_along_axis(device, index_dtype, axis):
    """``F.gather`` (``torch.gather``, ``np.take_along_axis``): each entry
    along the axis at its index, the others where they are; the indices no
    larger than the operand off the axis, of any integer dtype."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    shape = [3, 4, 5]
    shape[axis] = 7
    ids = rng.integers(0, x.shape[axis], shape).astype(index_dtype)
    got = lumen.compile(lambda x, i: F.gather(x, axis, i))(_tensor(x, device), _tensor(ids, device))
    want = np.take_along_axis(x[tuple(slice(n) if d != axis else slice(None) for d, n in enumerate(shape))], ids, axis)
    np.testing.assert_array_equal(lumen.to_numpy(got), want)


@pytest.mark.parametrize("device", DEVICES)
def test_take_along_axis_gradient(device):
    """Its gradient: each entry's cotangent summed over its reads, zeros
    where none read it."""
    rng = np.random.default_rng(6)
    x, c = rng.standard_normal((4, 3)).astype(np.float32), rng.standard_normal((5, 3)).astype(np.float32)
    ids = rng.integers(0, 4, (5, 3)).astype(np.int64)
    grad = lumen.grad(lambda x, i, c: F.sum(F.gather(x, 0, i) * c))
    got = lumen.compile(grad)(_tensor(x, device), _tensor(ids, device), _tensor(c, device))
    want = np.zeros_like(x)
    np.add.at(want, (ids, np.arange(3)[None, :]), c)
    np.testing.assert_allclose(lumen.to_numpy(got), want, rtol=1e-6, atol=1e-6)


def test_take_along_axis_checks_its_operands():
    """Indices are integers of the operand's rank, no larger off the axis."""
    m = lambda shape, dtype="float32": lumen.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, i: prims.take_along_axis(x, i, 1))(m([4, 6]), m([3, 9], "int8"))
    assert "f32[3,9] = take_along_axis[axis=1]" in str(g)
    for x, i in [([4, 6], [5, 2]), ([4, 6], [4]), ([4, 6], [4, 2, 1])]:
        with pytest.raises(ValueError, match="along axis 1"):
            lumen.make_graph(lambda x, i: prims.take_along_axis(x, i, 1))(m(x), m(i, "int32"))
    with pytest.raises(ValueError, match="along axis 1"):
        lumen.make_graph(lambda x, i: prims.take_along_axis(x, i, 1))(m([4, 6]), m([4, 6]))
