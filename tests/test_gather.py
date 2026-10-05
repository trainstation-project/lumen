"""Gathers (``prims.gather``, ``weight[ids]``: an embedding's lookup) and
scatter-adds (``prims.scatter_add``, a gather's gradient): entries along an
axis at integer indices read when the program runs, each clamped into the
axis, as XLA's gather (MLX's ``take``, ``scatter_add``)."""

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
def test_scatter_add_on_mps_takes_atomic_dtypes():
    """MPS adds float32, int32 and uint32 atomically: another dtype's
    scatter_add raises (a gather's gradient accumulates in float32)."""
    x = _tensor(np.zeros((4, 3), np.float32), "mps", "bfloat16")
    ids = _tensor(np.array([0, 2]), "mps")
    with pytest.raises(ValueError, match="atomically"):
        lumen.compile(lambda x, i: prims.scatter_add(x, i, x[i], 0))(x, ids)


def test_scatter_add_in_place():
    """Its value is its operand's memory where nothing reads the operand
    after it (a gradient's zeros): the updates added there alone."""
    args = [
        lumen.empty(s, dtype=d, device="meta") for s, d in (([4, 6], "float32"), ([3], "int64"), ([3, 6], "float32"))
    ]
    plan = str(lumen.graph.Plan(lumen.make_graph(lambda x, i, u: prims.scatter_add(x * 2.0, i, u, 0))(*args), "cpu"))
    assert "out0:f32[4,6] = scatter_add[axis=0] out0 " in plan, plan
