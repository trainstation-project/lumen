"""Gathers (``prims.gather``, ``weight[ids]``: an embedding's lookup) and
scatter-adds (``prims.scatter_add``, a gather's gradient): entries along an
axis at integer indices read when the program runs (MLX's ``take``,
``scatter_add``). Indices out of the axis here too, so with
``lumen.config.compiler.safe_kernels`` (each clamped into the axis, as
XLA's gather; the reference's always): without it, one out of range is the
program's error."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.mps)]


@pytest.fixture(autouse=True)
def safe_kernels():
    """Each test with its indices clamped (out of range here too)."""
    lumen.config.compiler.safe_kernels = True
    yield
    lumen.config.compiler.safe_kernels = False


def _tensor(a, device, dtype=None):
    try:
        t = lumen.from_numpy(a).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    return t if dtype is None else t.to(dtype=dtype)


# Repeated indices, and ones outside the axis (clamped).
IDS = np.array([[3, 0, 3], [1, 9, -2]])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
@pytest.mark.parametrize("axis", [0, 1])
def test_gather(device, index_dtype, dtype, axis):
    """The entries the indices pick along the axis (clamped into it), the
    axis replaced by the indices' shape, for any dtype; ``weight[ids]`` is
    one along the first."""
    x = (np.random.default_rng(0).standard_normal((4, 5, 3)) * 4).astype(np.float32)
    xt = _tensor(x, device, dtype)
    x = lumen.to_numpy(xt.to(dtype="float32"))
    ids = _tensor(IDS.astype(index_dtype), device)
    got = lumen.compile(lambda x, i: prims.gather(x, i, axis))(xt, ids)
    want = np.take(x, IDS, axis=axis, mode="clip")
    np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)
    if axis == 0:
        got = lumen.compile(lambda x, i: x[i])(xt, ids)
        np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)


@pytest.mark.mps
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_gather_fuses(index_dtype, axis):
    """A gather computes its elements in the kernel reading them (a loop
    fusion's, or a row kernel's: an embedding's lookup, its positions added,
    then normalized), reading the operand at each picked (clamped) index:
    no kernel of its own; the values the CPU's."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 5, 3)).astype(np.float32)
    xt, ids = _tensor(x, "mps"), _tensor(IDS.astype(index_dtype), "mps")

    def f(x, i):
        g = prims.gather(x * 2.0, i, axis) + 1.0
        return g, F.rms_norm(g, g.shape[-1])

    labels = [s["label"] for s in lumen.graph.Plan(lumen.make_graph(f)(xt, ids), "mps").steps()]
    assert "gather" not in labels and any("gather" in label for label in labels), labels
    got = lumen.compile(f)(xt, ids)
    want = lumen.compile(f, device="cpu")(lumen.from_numpy(x), lumen.from_numpy(IDS.astype(index_dtype)))
    for g, w in zip(got, want):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-5, atol=1e-6)


@pytest.mark.mps
def test_gather_of_a_table_of_as_many_elements_as_rows():
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
        h = (table[ids] + positions).reshape(1024, 64)
        return F.sum(F.rms_norm(h, 64) * c)

    grad = lumen.value_and_grad(loss, (0, 1))
    args = (table, positions, ids, c)
    got = lumen.compile(grad)(*(_tensor(a, "mps") for a in args))
    want = lumen.compile(grad, device="cpu")(*(lumen.from_numpy(a) for a in args))
    np.testing.assert_allclose(got[0].item(), want[0].item(), rtol=1e-5)
    for g, w in zip(got[1], want[1]):
        np.testing.assert_allclose(lumen.to_numpy(g), lumen.to_numpy(w), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "int32"])
@pytest.mark.parametrize("axis", [0, 1])
def test_scatter_add(device, dtype, axis):
    """Each update added at its index (clamped as the gather's), those at
    one entry summed."""
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((4, 5, 3)) * 4).astype(np.float32)
    u = (rng.standard_normal(np.take(x, IDS, axis=axis, mode="clip").shape) * 4).astype(np.float32)
    xt, ut = _tensor(x, device, dtype), _tensor(u, device, dtype)
    x, u = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, ut))
    ids = _tensor(IDS.astype(np.int64), device)
    got = lumen.compile(lambda x, i, u: prims.scatter_add(x, i, u, axis))(xt, ids, ut)
    want = x.copy()
    clipped = np.clip(IDS, 0, x.shape[axis] - 1)
    np.add.at(want, (slice(None),) * axis + (clipped,), u)
    np.testing.assert_allclose(lumen.to_numpy(got.to(dtype="float32")), want, rtol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gather_gradient(device, dtype):
    """A lookup's gradient: each row's cotangent summed over its reads (the
    clamped ones too), zeros where none read it, accumulated in float32."""
    rng = np.random.default_rng(2)
    w, c = rng.standard_normal((4, 3)).astype(np.float32), rng.standard_normal((2, 3, 3)).astype(np.float32)
    wt, ct = _tensor(w, device, dtype), _tensor(c, device, dtype)
    w, c = (lumen.to_numpy(t.to(dtype="float32")) for t in (wt, ct))
    ids = _tensor(IDS.astype(np.int64), device)
    grad = lumen.grad(lambda w, i, c: F.sum((w[i] * c).float()))
    got = lumen.compile(grad)(wt, ids, ct)
    want = np.zeros_like(w)
    np.add.at(want, np.clip(IDS, 0, 3), c)
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


@pytest.mark.mps
@pytest.mark.parametrize("width, rows", [(64, 5), (7, 5)])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
def test_scatter_add_on_mps_is_deterministic(dtype, width, rows):
    """On MPS a scatter_add adds each element's updates in index order (a
    thread an element scanning the indices, its SIMD group's lanes 32 at a
    time when they share a row: rows of 64; else one at a time: rows of 7;
    of many rows and updates, a thread a column; no atomics): the CPU's result bit for bit, for any dtype
    (bfloat16 too, each sum rounded as the reference rounds it), the same
    each run; a lookup's gradient too (deterministic training)."""
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((rows, width)) * 4).astype(np.float32)
    ids = rng.integers(0, rows, (300,))
    u = (rng.standard_normal((300, width)) * 4).astype(np.float32)
    f = lambda x, i, u: prims.scatter_add(x, i, u, 0)  # noqa: E731
    on = lambda device: [
        _tensor(a, device, dtype) if a.dtype == np.float32 else _tensor(a, device) for a in (x, ids, u)
    ]  # noqa: E731
    want = lumen.to_numpy(lumen.compile(f, device="cpu")(*on("cpu")).to(dtype="float32"))
    mps = lumen.compile(f, device="mps")
    for _ in range(3):
        got = lumen.to_numpy(mps(*on("mps")).to(dtype="float32"))
        np.testing.assert_array_equal(got, want)


def test_scatter_add_in_place():
    """Its value is its operand's memory where nothing reads the operand
    after it (a gradient's zeros): the updates added there alone."""
    args = [
        lumen.empty(s, dtype=d, device="meta") for s, d in (([4, 6], "float32"), ([3], "int64"), ([3, 6], "float32"))
    ]
    plan = str(lumen.graph.Plan(lumen.make_graph(lambda x, i, u: prims.scatter_add(x * 2.0, i, u, 0))(*args), "cpu"))
    assert "out0:f32[4,6] = scatter_add[axis=0] out0 " in plan, plan
