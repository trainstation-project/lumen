"""Dynamic slices (``lax.dynamic_slice``, ``lax.dynamic_update_slice``):
blocks at start indices read when the program runs, clamped so the block is
inside, their gradients, and a dynamic_update_slice written in place
(``graph/plan.rs``): into its operand's memory where nothing reads the
operand after it, a donated input's or a weight assigned with ``copy_``
too (a KV cache), copying only where something does."""

import numpy as np
import pytest

import lumen
import lumen.functional as F
from lumen import prims

DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.mps)]


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
    return str(lumen.graph.Plan(lumen.make_graph(f)(*args), "cpu", **options))


def test_dynamic_update_slice_in_place():
    """Its value is its operand's memory, the update written there alone,
    where nothing reads the operand after it: a value it computes, or a
    donated input; where something does, its operand is copied first."""
    f32, i32 = "float32", "int32"
    shapes = (([4, 6], f32), ([1, 6], f32), ([], i32))
    # x * 2 computed into the output, then updated there.
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x * 2.0, u, (i, 0)), *shapes)
    assert "out0:f32[4,6] = dynamic_update_slice out0 " in plan, plan
    # x * 2 read later: copied, kept.
    plan = _plan(lambda x, u, i: (prims.dynamic_update_slice(x * 2.0, u, (i, 0)), x * 2.0 + 1.0), *shapes)
    assert "out0:f32[4,6] = dynamic_update_slice ws+" in plan, plan
    # An input: in place only donated.
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x, u, (i, 0)), *shapes)
    assert "out0:f32[4,6] = dynamic_update_slice in0 " in plan, plan
    plan = _plan(lambda x, u, i: prims.dynamic_update_slice(x, u, (i, 0)), *shapes, donate=[0])
    assert "in0:f32[4,6] = dynamic_update_slice in0 " in plan, plan


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
    plan = str(lumen.graph.Plan(graph, device, donate=[2], parameters=[2]))
    assert "in2:f32[1,64,4] = dynamic_update_slice in2 " in plan, plan
    memory = cache.k._placed(device)
    for t in range(5):
        f(cache, _tensor(np.full((1, 1, 4), t + 1.0, np.float32), device), _tensor(np.array([t], np.int32), device))
    assert cache.k._placed(device).storage_id == memory.storage_id
    k = lumen.to_numpy(cache.k._placed(device))
    np.testing.assert_array_equal(k[0, :6, 0], [1, 2, 3, 4, 5, 0])
