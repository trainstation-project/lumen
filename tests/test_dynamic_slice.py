import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

DEVICES = [
    "cpu",
    pytest.param("mps", marks=pytest.mark.mps),
    pytest.param("cuda", marks=pytest.mark.cuda),
    pytest.param("tpu", marks=pytest.mark.tpu),
]


@pytest.fixture(autouse=True)
def triton():
    """CUDA's programs on the Triton backend."""
    lumen.config.compiler.cuda.use_triton_backend = True
    yield
    lumen.config.compiler.cuda.use_triton_backend = False


def _tensor(a, device, dtype=None):
    try:
        t = lumen.from_numpy(a).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))
    return t if dtype is None else t.to(dtype=dtype)


def _block(shape, starts, sizes):
    """The block's slices, its starts clamped as XLA's."""
    starts = [min(max(s, 0), n - k) for s, n, k in zip(starts, shape, sizes)]
    return tuple(slice(s, s + k) for s, k in zip(starts, sizes))


def _scalars(*indices):
    return [i.reshape(()) for i in indices]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "int32"])
@pytest.mark.parametrize("starts", [(1, 4, 0), (3, 9, 0), (-2, -1, 5)], ids=["inside", "clamped", "negative"])
def test_dynamic_slices(device, index_dtype, dtype, starts):
    """Each reads (writes) the block at its starts, clamped inside the
    operand, for any dtype and int32 or int64 indices."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, 10, 3)) * 4).astype(np.float32)
    u = (rng.standard_normal((2, 3, 3)) * 4).astype(np.float32)
    xt, ut = _tensor(x, device, dtype), _tensor(u, device, dtype)
    x, u = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, ut))
    idx = [_tensor(np.array([s], index_dtype), device) for s in starts]
    block = _block(x.shape, starts, u.shape)
    ds = lumen.compile(lambda x, *i: prims.dynamic_slice(x, _scalars(*i), u.shape))(xt, *idx)
    np.testing.assert_array_equal(lumen.to_numpy(ds.to(dtype="float32")), x[block])
    dus = lumen.compile(lambda x, u, *i: prims.dynamic_update_slice(x, u, _scalars(*i)))(xt, ut, *idx)
    want = x.copy()
    want[block] = u
    np.testing.assert_array_equal(lumen.to_numpy(dus.to(dtype="float32")), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "shape, update, starts",
    [
        ((512, 1024), (64, 1024), (64, 0)),
        ((512, 1024), (64, 1024), (37, 0)),
        ((300, 700), (40, 300), (13, 77)),
        ((2048, 2048), (1504, 1300), (7, 700)),
        ((3, 70, 260), (2, 9, 130), (1, 5, 3)),
        ((4, 64, 256), (1, 64, 256), (2, 0, 0)),
    ],
    ids=["aligned", "rows", "unaligned", "blocks", "lead", "slab"],
)
@pytest.mark.parametrize("in_place", [True, False])
def test_dynamic_update_slices(device, dtype, shape, update, starts, in_place):
    """The update lands at its (clamped) starts, the rest of the operand
    kept, whether it is written into the operand's memory (an intermediate)
    or into a copy of it (an argument)."""
    rng = np.random.default_rng(0)
    xt = _tensor(rng.standard_normal(shape).astype(np.float32), device, dtype)
    ut = _tensor(rng.standard_normal(update).astype(np.float32), device, dtype)
    x, u = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, ut))
    idx = [_tensor(np.array([s], np.int32), device) for s in starts]
    operand = (lambda x: x * 2.0) if in_place else (lambda x: x)
    got = lumen.compile(lambda x, u, *i: prims.dynamic_update_slice(operand(x), u, _scalars(*i)))(xt, ut, *idx)
    want = 2 * x if in_place else x.copy()
    want[_block(shape, starts, update)] = u
    np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "shape, update, starts",
    [
        ((512, 256), (32, 256), [(0, 0), (32, 0), (96, 0), (480, 0)]),
        ((512, 256), (64, 256), [(64, 0), (128, 0), (192, 0), (256, 0)]),
        ((300, 700), (40, 300), [(13, 77), (0, 0), (290, 500), (64, 128)]),
        ((3, 70, 260), (1, 9, 130), [(1, 5, 3), (0, 0, 0), (2, 61, 130)]),
    ],
    ids=["aligned", "tiled", "unaligned", "lead"],
)
@pytest.mark.parametrize("in_place", [True, False])
def test_chained_dynamic_update_slices(device, dtype, shape, update, starts, in_place):
    """Updates at constant starts, each into the last's result, land in
    order (later ones over earlier ones); on TPU they run as one step."""
    rng = np.random.default_rng(1)
    xt = _tensor(rng.standard_normal(shape).astype(np.float32), device, dtype)
    uts = [_tensor(rng.standard_normal(update).astype(np.float32), device, dtype) for _ in starts]
    x, *us = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, *uts))

    def f(x, *us):
        x = x * 2.0 if in_place else x
        for u, at in zip(us, starts):
            x = prims.dynamic_update_slice(x, u, at)
        return x

    if device == "tpu":
        metas = [lumen.empty(list(t.shape), dtype, "meta") for t in (xt, *uts)]
        steps = lumen.graph_compiler.Plan(lumen.make_graph(f)(*metas), "tpu").steps()
        assert [s["label"] for s in steps].count(f"{len(starts)}x dynamic_update_slice") == 1, steps
    got = lumen.compile(f)(xt, *uts)
    want = 2 * x if in_place else x.copy()
    for u, at in zip(us, starts):
        want[_block(shape, at, update)] = u
    np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("start", [3, 9, -2], ids=["inside", "clamped", "negative"])
def test_narrow_at_tensor_start(device, dtype, start):
    """``narrow`` at a 0-d tensor start reads the block there, and a
    ``copy_`` into it writes the block in place, its start clamped as
    ``prims.dynamic_slice``'s."""
    rng = np.random.default_rng(0)
    xt = _tensor(rng.standard_normal((2, 10, 3)).astype(np.float32), device, dtype)
    ut = _tensor(rng.standard_normal((2, 4, 3)).astype(np.float32), device, dtype)
    x, u = (lumen.to_numpy(t.to(dtype="float32")) for t in (xt, ut))
    i = _tensor(np.array([start], np.int32), device)
    block = _block(x.shape, (0, start, 0), u.shape)

    read = lumen.compile(lambda x, i: x.narrow(1, i.reshape(()), 4) * 2.0)(xt, i)
    np.testing.assert_array_equal(lumen.to_numpy(read.to(dtype="float32")), 2 * x[block])

    def write(x, u, i):
        y = x * 2.0
        y.narrow(1, i.reshape(()), 4).copy_(u)
        return y

    want = 2 * x
    want[block] = u
    got = lumen.compile(write)(xt, ut, i)
    np.testing.assert_array_equal(lumen.to_numpy(got.to(dtype="float32")), want)


def test_start_indices():
    """Python ints are constants of the traced indices' dtype; indices must
    be int32 or int64 scalars of one dtype, and the block inside."""
    m = lambda shape, dtype="float32": lumen.empty(shape, dtype=dtype, device="meta")  # noqa: E731
    g = lumen.make_graph(lambda x, i: prims.dynamic_slice(x, (i, 2), (1, 3)))(m([4, 6]), m([], "int64"))
    assert "dynamic_slice[slice_sizes=(1, 3)]" in str(g) and "dtype=i64" in str(g)
    with pytest.raises(ValueError, match="int32 or int64"):
        lumen.make_graph(lambda x, i: prims.dynamic_slice(x, (i, i), (1, 3)))(m([4, 6]), m([], "float32"))
    with pytest.raises(ValueError, match="int32 or int64"):
        lumen.make_graph(lambda x, i, j: prims.dynamic_slice(x, (i, j), (1, 3)))(
            m([4, 6]), m([], "int32"), m([], "int64")
        )
    with pytest.raises(ValueError, match="block"):
        lumen.make_graph(lambda x: prims.dynamic_slice(x, (0, 0), (5, 3)))(m([4, 6]))
    with pytest.raises(ValueError, match="block"):
        lumen.make_graph(lambda x, u: prims.dynamic_update_slice(x, u, (0, 0)))(m([4, 6]), m([2, 7]))


@pytest.mark.parametrize("device", DEVICES)
def test_dynamic_slice_gradients(device):
    """Linear in the operand and update: a slice's gradient is its
    cotangent in the block, zeros elsewhere; an update's, the block of the
    cotangent, and its operand's, the cotangent but in the block."""
    rng = np.random.default_rng(1)
    x, u, w = (rng.standard_normal(s).astype(np.float32) for s in ((5, 6), (2, 3), (5, 6)))
    xt, ut, wt = (_tensor(a, device) for a in (x, u, w))
    i = _tensor(np.array([2], np.int32), device)
    block = _block(x.shape, (2, 1), u.shape)

    def sliced(x, w, i):
        return F.sum(
            prims.dynamic_slice(x, (i.reshape(()), 1), (2, 3)) * prims.dynamic_slice(w, (i.reshape(()), 1), (2, 3))
        )

    gx = lumen.compile(lumen.grad(sliced, 0))(xt, wt, i)
    want = np.zeros_like(x)
    want[block] = w[block]
    np.testing.assert_allclose(lumen.to_numpy(gx), want)

    def updated(x, u, w, i):
        return F.sum(prims.dynamic_update_slice(x, u, (i.reshape(()), 1)) * w)

    gx, gu = lumen.compile(lumen.grad(updated, (0, 1)))(xt, ut, wt, i)
    want = w.copy()
    want[block] = 0
    np.testing.assert_allclose(lumen.to_numpy(gx), want)
    np.testing.assert_allclose(lumen.to_numpy(gu), w[block])


def _plan(f, *shapes, **options):
    args = [lumen.empty(s, dtype=d, device="meta") for s, d in shapes]
    return str(lumen.graph_compiler.Plan(lumen.make_graph(f)(*args), "cpu", **options))


def _in_place(plan):
    """Whether the plan's dynamic_update_slice writes its operand's memory."""
    (line,) = [line for line in plan.splitlines() if "dynamic_update_slice" in line]
    return line.split(":")[0].strip() == line.split("dynamic_update_slice ")[1].split()[0]


def test_dynamic_update_slice_in_place():
    """Its value is its operand's memory, the update written there alone,
    where nothing reads the operand after it: a value it computes, or a
    donated input; where something does, its operand is copied first."""
    f32, i32 = "float32", "int32"
    shapes = (([4, 6], f32), ([1, 6], f32), ([], i32))
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x * 2.0, u, (i, 0)), *shapes)
    assert "out0:f32[4,6] = dynamic_update_slice out0 " in plan, plan
    plan = _plan(lambda x, u, i: (prims.dynamic_update_slice(x * 2.0, u, (i, 0)), x * 2.0 + 1.0), *shapes)
    assert not _in_place(plan), plan
    plan = _plan(
        lambda x, u, i: (lambda e: (prims.dynamic_update_slice(e, u, (i, 0)), e.reshape(24) * 3.0))(x * 2.0), *shapes
    )
    assert not _in_place(plan), plan
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x, u, (i, 0)), *shapes)
    assert "out0:f32[4,6] = dynamic_update_slice in0 " in plan, plan
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x, u, (i, 0)), *shapes, donate=[0])
    assert "in0:f32[4,6] = dynamic_update_slice in0 " in plan, plan


@pytest.mark.parametrize("device", DEVICES)
def test_operand_read_after_its_update(device):
    """An operand (or a reshape of it) read after its update keeps its old
    value: the update is written into a copy."""
    rng = np.random.default_rng(2)
    x, u = rng.standard_normal((4, 6)).astype(np.float32), rng.standard_normal((1, 6)).astype(np.float32)
    xt, ut = _tensor(x, device), _tensor(u, device)
    i = _tensor(np.array([2], np.int32), device)

    def f(x, u, i):
        e = x * 2.0
        return prims.dynamic_update_slice(e, u, (i.reshape(()), 0)), e.reshape(24) * 3.0

    updated, later = (lumen.to_numpy(t) for t in lumen.compile(f)(xt, ut, i))
    want = 2 * x
    want[2] = u[0]
    np.testing.assert_allclose(updated, want, rtol=1e-6)
    np.testing.assert_allclose(later, 6 * x.reshape(24), rtol=1e-6)


class _Cache(lumen.nn.Module):
    k: lumen.Tensor


@pytest.mark.parametrize("device", DEVICES)
def test_kv_cache_is_written_in_place(device):
    """A decode step writing its token's keys into a module's cache
    (``c.k.copy_(dynamic_update_slice(c.k, k, (0, t, 0)))``): the update is
    written into the cache's memory, nothing copied (its memory is donated),
    and each step's position holds its token."""
    cache = _Cache(lumen.empty([1, 64, 4], device="meta"))

    def step(c, k, t):
        c.k.copy_(prims.dynamic_update_slice(c.k, k, (0, t.reshape(()), 0)))
        return F.sum(k)

    f = lumen.compile(step, device=device)
    meta = (lumen.empty([1, 1, 4], device="meta"), lumen.empty([1], dtype="int32", device="meta"))
    f(cache, *meta)
    graph = lumen.make_graph(step)(cache, *meta)
    plan = str(lumen.graph_compiler.Plan(graph, device, donate=[2], parameters=[2]))
    assert "in2:f32[1,64,4] = dynamic_update_slice in2 " in plan, plan
    memory = cache.k._placed(device)
    for t in range(5):
        f(cache, _tensor(np.full((1, 1, 4), t + 1.0, np.float32), device), _tensor(np.array([t], np.int32), device))
    assert cache.k._placed(device).storage_id == memory.storage_id
    k = lumen.to_numpy(cache.k._placed(device))
    np.testing.assert_array_equal(k[0, :6, 0], [1, 2, 3, 4, 5, 0])
