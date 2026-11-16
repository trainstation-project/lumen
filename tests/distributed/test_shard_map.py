"""Tests of lumen.distributed's shard_map: partition specs on 1-D and 2-D meshes, the named collectives, their
gradients and errors, in one process and across CPU processes talking through the store."""

import numpy as np
import pytest
from helpers import launch

import lumen
import lumen.functional as F
from lumen.distributed import P, PartitionSpec, init_device_mesh, shard_map, spmd


def test_partition_spec_is_a_tuple_of_its_entries():
    """A ``PartitionSpec`` (``P``) is the tuple of its entries and reprs as constructed."""
    spec = P("dp", None, ("dp", "tp"))
    assert spec == ("dp", None, ("dp", "tp")) and isinstance(spec, PartitionSpec)
    assert repr(spec) == "PartitionSpec('dp', None, ('dp', 'tp'))" and P() == ()


def test_a_mesh_of_one_process():
    """On a mesh of one process the mapped function sees the whole tensors, every collective is its operand
    (stacked along a new axis where JAX's would), the axis index is 0, and gradients flow through a decorated
    function."""
    mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    x = np.arange(12, dtype=np.float32).reshape(3, 4)

    def f(a):
        """Every named collective of ``a`` along ``tp`` or both axes."""
        return (
            spmd.psum(a, "tp") + spmd.axis_index(("dp", "tp")) + spmd.axis_size("dp"),
            spmd.pmean(a, ("dp", "tp")),
            spmd.pmax(a, "dp"),
            spmd.pmin(a, "tp"),
            spmd.all_gather(a, "tp", axis=1),
            spmd.all_gather(a, "tp", axis=1, tiled=True),
            spmd.psum_scatter(a.reshape(1, *a.shape), "dp"),
            spmd.all_to_all(a.reshape(1, *a.shape), "tp", 0, 1),
            spmd.ppermute(a, "tp", [(0, 0)]),
        )

    mapped = shard_map(f, mesh=mesh, in_specs=P("dp", "tp"), out_specs=P("dp", "tp"))
    got = [lumen.to_numpy(t) for t in lumen.compile(mapped, device="cpu")(lumen.from_numpy(x))]
    expected = [x + 1, x, x, x, x[:, None], x, x, x[:, None], x]
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, e)

    @shard_map(mesh=mesh, in_specs=P("dp"), out_specs=P("dp"))
    def square(b):
        """``b`` squared, mapped as a decorator."""
        return b * b

    grad = lumen.compile(lumen.grad(lambda a: F.sum(square(a))), device="cpu")(lumen.from_numpy(x))
    np.testing.assert_array_equal(lumen.to_numpy(grad), 2 * x)


def test_unknown_axis_names_are_errors():
    """A named collective outside a shard_map, or naming an axis its mesh doesn't have, and a spec naming an
    unknown axis or too many dimensions, raise naming the axis."""
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("x",))
    x = lumen.from_numpy(np.ones((2, 3), np.float32))
    with pytest.raises(NameError, match="psum\\(\\): unbound axis name 'x': named collectives run inside"):
        lumen.compile(lambda a: spmd.psum(a, "x"), device="cpu")(x)
    with pytest.raises(NameError, match="all_gather\\(\\): unbound axis name 'y'; the shard_map's mesh has axes"):
        lumen.compile(shard_map(lambda a: spmd.all_gather(a, "y"), mesh=mesh, in_specs=P(), out_specs=P()))(x)
    with pytest.raises(NameError, match="axis_index\\(\\): unbound axis name 'y'"):
        lumen.compile(shard_map(lambda a: a + spmd.axis_index(("x", "y")), mesh=mesh, in_specs=P(), out_specs=P()))(x)
    with pytest.raises(NameError, match="in_specs PartitionSpec\\('z'\\) names axis 'z'"):
        lumen.compile(shard_map(lambda a: a, mesh=mesh, in_specs=P("z"), out_specs=P()))(x)
    with pytest.raises(ValueError, match="more entries than the tensor's 2 dimensions"):
        lumen.compile(shard_map(lambda a: a, mesh=mesh, in_specs=P(), out_specs=P(None, None, "x")))(x)


# Run in each process: shard_maps over a 1-D mesh of every process ("x"), and with 4 processes a 2x2 mesh
# ("dp", "tp"), each returning its global outputs; values and gradients go back as JSON.
SHARD_MAP_PROGRAM = """
import json
import numpy as np
import lumen
import lumen.distributed as dist
import lumen.functional as F
from lumen.distributed import P, shard_map, spmd

dist.init_process_group()
rank, n = dist.get_rank(), dist.get_world_size()
mesh = dist.init_device_mesh("cpu", (n,), mesh_dim_names=("x",))


def host(out):
    \"\"\"The tensors in ``out`` (nested tuples of them), flattened in order, as lists.\"\"\"
    if isinstance(out, (tuple, list)):
        return [v for t in out for v in host(t)]
    return [lumen.to_numpy(out).tolist()]


run = lambda f, *args: host(lumen.compile(f, device="cpu")(*(lumen.from_numpy(a) for a in args)))
smap = lambda f, i, o, m=mesh: shard_map(f, mesh=m, in_specs=i, out_specs=o)
A = np.arange(4 * n * 3, dtype=np.float32).reshape(4 * n, 3)
B = np.arange(n * n * 2 * n, dtype=np.float32).reshape(n * n, 2 * n) - 7
W = np.arange(n * n * 3, dtype=np.float32).reshape(n, n, 3)
mm = lambda x, y: F.matmul(x, y, "float32", "float32")
ring = [(i, (i + 1) % n) for i in range(n)]
out = {"rank": rank}

# In and out specs: sharded rows, sharded columns, replicated in and out, and a function of several outputs.
out["specs"] = run(lambda a, b: (
    smap(lambda a: a * 2 + spmd.axis_index("x"), P("x"), P("x"))(a),
    smap(lambda b: b + 1, P(None, "x"), P(None, "x"))(b),
    smap(lambda a: spmd.psum(a, "x"), P(), P())(a),
    *smap(lambda a, b: (a, b * 0 + spmd.axis_size("x")), (P("x"), P()), (P("x"), P()))(a, b),
), A, B)

# Every named collective, on blocks of the rows of A (or of B, W).
out["collectives"] = run(lambda a, b, w: (
    smap(lambda a: (spmd.psum(a, "x"), spmd.pmean(a, "x"), spmd.pmax(a, "x"), spmd.pmin(a, "x")), P("x"), P())(a),
    smap(lambda a: (spmd.all_gather(a, "x", tiled=True), spmd.all_gather(a, "x", axis=1)), P("x"), P())(a),
    smap(lambda b: spmd.psum_scatter(b, "x", scatter_dimension=1, tiled=True), P(), P(None, "x"))(b),
    smap(lambda w: spmd.psum_scatter(w[0], "x"), P(), P("x"))(w),
    smap(lambda b: spmd.all_to_all(b, "x", 1, 0, tiled=True), P("x"), P(None, "x"))(b),
    smap(lambda w: spmd.all_to_all(w, "x", 1, 0), P("x"), P("x"))(w),
    smap(lambda a: spmd.ppermute(a, "x", ring), P("x"), P("x"))(a),
    smap(lambda a: spmd.ppermute(a, "x", [(0, n - 1)]), P("x"), P("x"))(a),
), A, B, W)

# Tensor parallelism: a column-parallel matmul, relu, a row-parallel matmul and a psum, then its gradients.
X = np.linspace(-1, 1, 4 * 8, dtype=np.float32).reshape(4, 8)
W1 = np.linspace(-1, 1, 8 * 4 * n, dtype=np.float32).reshape(8, 4 * n)
W2 = np.linspace(1, -1, 4 * n * 6, dtype=np.float32).reshape(4 * n, 6)
C = np.linspace(0, 2, 4 * 6, dtype=np.float32).reshape(4, 6)
mlp = smap(lambda x, w1, w2: spmd.psum(mm(F.relu(mm(x, w1)), w2), "x"), (P(), P(None, "x"), P("x", None)), P())
out["mlp"] = run(mlp, X, W1, W2)
loss = lambda x, w1, w2, c: F.sum(mlp(x, w1, w2) * c)
out["mlp_grads"] = run(lumen.grad(loss, (0, 1, 2)), X, W1, W2, C)

# Gradients through sharded inputs and outputs, an all_gather and a ppermute.
D = np.linspace(-1, 1, 4 * n * 3, dtype=np.float32).reshape(4 * n, 3)
out["grads"] = run(lumen.grad(lambda a, d: (
    F.sum(smap(lambda a: a * a, P("x"), P("x"))(a) * d)
    + F.sum(smap(lambda a: spmd.all_gather(a, "x", tiled=True), P("x"), P())(a) * d)
    + F.sum(smap(lambda a: spmd.ppermute(a, "x", ring), P("x"), P("x"))(a) * d)
)), A, D)

if n == 4:
    grid = dist.init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    G = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    H = np.arange(2 * 3, dtype=np.float32).reshape(2, 3) + 1
    flat = np.arange(8, dtype=np.float32)
    index = lambda a: a * 0 + spmd.axis_index(("dp", "tp"))
    out["grid"] = run(lambda g, flat: (
        smap(lambda g: g + 10 * spmd.axis_index("dp") + spmd.axis_index("tp"), P("dp", "tp"), P("dp", "tp"), grid)(g),
        smap(lambda g: g + spmd.axis_index("tp"), P("tp"), P("tp"), grid)(g),
        smap(index, P(("dp", "tp")), P(("dp", "tp")), grid)(flat),
        smap(index, P(("tp", "dp")), P(("tp", "dp")), grid)(flat),
        smap(index, P(("dp", "tp")), P(("tp", "dp")), grid)(flat),
        smap(lambda g: spmd.psum(g, "tp"), P("dp", "tp"), P("dp"), grid)(g),
        smap(lambda g: spmd.psum(g, ("dp", "tp")), P("dp", "tp"), P(), grid)(g),
        smap(lambda g: spmd.all_gather(g, "dp", tiled=True), P("dp", "tp"), P(None, "tp"), grid)(g),
    ), G, flat)
    out["grid_grads"] = run(lumen.grad(lambda g, h: (
        F.sum(smap(lambda g: spmd.psum(g * g, "tp"), P("dp", "tp"), P("dp"), grid)(g))
        + F.sum(smap(lambda g, h: g * h, (P("dp", "tp"), P()), P("dp", "tp"), grid)(g, h))
    ), (0, 1)), G, H)
print(json.dumps(out))
"""


def relu_mlp_grads(x, w1, w2, c):
    """Gradients of ``sum(relu(x @ w1) @ w2 * c)`` with respect to ``x``, ``w1`` and ``w2``."""
    h = x @ w1
    dy = c
    dw2 = np.maximum(h, 0).T @ dy
    dh = (dy @ w2.T) * (h > 0)
    return dh @ w1.T, x.T @ dh, dw2


@pytest.mark.parametrize("world_size", [2, 4])
def test_shard_map_across_processes_matches_numpy(world_size):
    """A process per CPU (talking through the store) maps functions over a 1-D mesh, and with 4 processes a
    2x2 one: sharded, replicated and tuple-axis specs give the global outputs; each named collective, the axis
    index and size match NumPy; a column- then row-parallel MLP matches the unsharded one, and so do the
    gradients through it, through sharded inputs and outputs, an all_gather, a ppermute and inputs replicated
    along an axis."""
    n = world_size
    results = launch(SHARD_MAP_PROGRAM, n)
    A = np.arange(4 * n * 3, dtype=np.float32).reshape(4 * n, 3)
    B = np.arange(n * n * 2 * n, dtype=np.float32).reshape(n * n, 2 * n) - 7
    W = np.arange(n * n * 3, dtype=np.float32).reshape(n, n, 3)
    blocks = np.split(A, n)
    X = np.linspace(-1, 1, 4 * 8, dtype=np.float32).reshape(4, 8)
    W1 = np.linspace(-1, 1, 8 * 4 * n, dtype=np.float32).reshape(8, 4 * n)
    W2 = np.linspace(1, -1, 4 * n * 6, dtype=np.float32).reshape(4 * n, 6)
    C = np.linspace(0, 2, 4 * 6, dtype=np.float32).reshape(4, 6)
    D = np.linspace(-1, 1, 4 * n * 3, dtype=np.float32).reshape(4 * n, 3)
    first = np.concatenate([blocks[0] if i == n - 1 else np.zeros_like(blocks[0]) for i in range(n)])
    expected = {
        "specs": [A * 2 + np.repeat(np.arange(n), 4)[:, None], B + 1, n * A, A, np.full_like(B, n)],
        "collectives": [
            sum(blocks),
            sum(blocks) / n,
            np.max(blocks, 0),
            np.min(blocks, 0),
            A,
            np.stack(blocks, 1),
            n * B,
            n * W[0].reshape(-1),
            B,
            W.transpose(1, 0, 2).reshape(n * n, 1, 3),
            np.roll(A, 4, 0),
            first,
        ],
        "mlp": [np.maximum(X @ W1, 0) @ W2],
        "mlp_grads": relu_mlp_grads(X, W1, W2, C),
        "grads": [2 * A * D + D + np.roll(D, -4, 0)],
    }
    if n == 4:
        G = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
        H = np.arange(2 * 3, dtype=np.float32).reshape(2, 3) + 1
        dp, tp = np.repeat([0, 1], 2)[:, None], np.repeat([0, 1], 3)[None, :]
        # The rows of each (dp, tp) block of G, as the 2x2 grid of 2x3 blocks.
        grid = G.reshape(2, 2, 2, 3).transpose(0, 2, 1, 3)
        expected["grid"] = [
            G + 10 * dp + tp,
            G + np.repeat([0, 1], 2)[:, None],
            np.repeat(np.arange(4, dtype=np.float32), 2),
            np.repeat([0, 2, 1, 3], 2).astype(np.float32),
            np.repeat([0, 2, 1, 3], 2).astype(np.float32),
            G.reshape(4, 2, 3).sum(1),
            grid.sum((0, 1)),
            G,
        ]
        expected["grid_grads"] = [2 * G + np.tile(H, (2, 2)), grid.sum((0, 1))]
    for rank, r in enumerate(results):
        for key, values in expected.items():
            assert len(r[key]) == len(values), key
            for i, (g, e) in enumerate(zip(r[key], values, strict=True)):
                np.testing.assert_allclose(np.array(g), e, rtol=1e-6, atol=1e-5, err_msg=f"rank {rank} {key}[{i}]")
