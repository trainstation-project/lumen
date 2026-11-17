"""Tests of sharding placements (``Shard``, ``Replicate``, ``Partial``) on ``lumen.Tensor`` and traced tensors: the
API on a mesh of one device, and ops, gradients and redistributions across CPU processes against NumPy."""

import numpy as np
import pytest
from helpers import CPU, CUDA, MPS, TRITON_CUDA, launch, rand

import lumen
import lumen.distributed as dist
import lumen.functional as F
from lumen.distributed import Partial, Replicate, Shard, from_local
from lumen.tensor.placements import Sharding, sharding

# Environment variables a launcher like torchrun sets for each process.
LAUNCHER = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def one_process(monkeypatch):
    """Runs every test as a single process, without the launcher's variables."""
    for name in LAUNCHER:
        monkeypatch.delenv(name, raising=False)


def mm(x, w):
    """``x @ w`` in float32."""
    return F.matmul(x, w, "float32", "float32")


def mlp(x, w1, w2):
    """A two-layer MLP, ``tanh(x @ w1) @ w2``: column-parallel then row-parallel when ``w1`` is sharded along
    its columns and ``w2`` along its rows (Megatron's)."""
    return mm(F.tanh(mm(x, w1)), w2)


def test_placements_resolve_against_the_tensor():
    """A Shard takes a position or an axis name, resolved to a position (one dimension may be sharded over
    several mesh dimensions); malformed placements raise naming what is wrong."""
    mesh = dist.init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    spec = sharding(mesh, [Shard("seq"), Shard(-1)], 3, ("batch", "seq", "hidden"))
    assert spec == Sharding(mesh, (Shard(1), Shard(2)))
    with pytest.raises(ValueError, match="1 placements"):
        sharding(mesh, [Replicate()], 2, None)
    assert sharding(mesh, [Shard(0), Shard(0)], 2, None).placements == (Shard(0), Shard(0))
    with pytest.raises(ValueError, match="reduce_op"):
        sharding(mesh, [Partial("prod"), Replicate()], 2, None)
    with pytest.raises(KeyError, match="seq"):
        sharding(mesh, [Shard("seq"), Replicate()], 2, None)
    with pytest.raises(TypeError, match="placement"):
        sharding(mesh, ["shard", Replicate()], 2, None)


def test_a_sharded_tensor_on_one_device():
    """On a 1-device mesh, a distributed tensor has its global shape, placements and mesh, shares memory with
    its local shard, keeps its names, crosses lumen.compile placed (a placement of its own signature), and
    refuses what would read its local shard as global: views and NumPy. Collectives run inside a trace only."""
    mesh = dist.init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    a = rand(4, 6)
    x = lumen.from_numpy(a).rename("seq", "hidden").distribute(mesh, [Shard("hidden")])
    assert (x.shape, x.placements, x.device_mesh, x.names) == ([4, 6], (Shard(1),), mesh, ("seq", "hidden"))
    assert "placement=Sharding([Shard(dim=1)]" in repr(x)
    local = x.to_local()
    assert local.placement is None and local.shares_storage_with(x) and local.names == ("seq", "hidden")
    np.testing.assert_array_equal(lumen.to_numpy(local.rename(None)), a)
    back = from_local(local, mesh, [Partial()])
    assert (back.placements, back.shape) == ((Partial(),), [4, 6])
    with pytest.raises(TypeError, match="to_local"):
        x.reshape([24])
    with pytest.raises(TypeError, match="to_local"):
        lumen.to_numpy(x)
    with pytest.raises(RuntimeError, match="only while tracing"):
        x.full_tensor()
    f = lumen.compile(lambda x: x * 2.0)
    y, plain = f(x), f(local)
    assert (y.placements, y.names, plain.placement) == ((Shard(1),), ("seq", "hidden"), None)
    assert x._signature != local._signature
    np.testing.assert_array_equal(lumen.to_numpy(y.to_local().rename(None)), 2 * a)


@pytest.mark.parametrize("device", [CPU, MPS, CUDA, TRITON_CUDA], indirect=True)
def test_placements_on_one_device_keep_results(device):
    """On a 1-device mesh, sharding the weights of an MLP column- then row-parallel changes neither its value
    nor its gradients, which come back placed as the weights; the inputs' strategies keep the shards."""
    mesh = dist.init_device_mesh(device, (1,), mesh_dim_names=("tp",))
    x, w1, w2 = (lumen.from_numpy(rand(*s, seed=k)).to(device) for k, s in enumerate([(4, 8), (8, 16), (16, 8)]))
    loss = lambda x, w1, w2: F.sum(F.softmax(mlp(x, w1, w2), 1) * x)  # noqa: E731
    f = lumen.compile(lambda x, w1, w2: (loss(x, w1, w2), *lumen.grad(loss, (1, 2))(x, w1, w2)))
    plain = [lumen.to_numpy(t.to("cpu")) for t in f(x, w1, w2)]
    sharded = f(x.distribute(mesh, [Replicate()]), w1.distribute(mesh, [Shard(1)]), w2.distribute(mesh, [Shard(0)]))
    assert [t.placements for t in sharded] == [(Replicate(),), (Shard(1),), (Shard(0),)]
    full = lumen.compile(lambda *ts: tuple(t.full_tensor() for t in ts))
    for got, want in zip(full(*sharded), plain, strict=True):
        np.testing.assert_array_equal(lumen.to_numpy(got.to("cpu")), want)


# Run in each process: a mesh over the world (and, with 4 processes, a 2x2 dp x tp mesh). A tensor every process
# holds is distributed and redistributed between every pair of placements; elementwise ops and reductions run on
# mixed placements; a column- then row-parallel MLP (a lumen.nn.Module of sharded weights) takes an SGD step from
# backward(); a named tensor is sharded by axis name; and the collectives traced for tensor parallelism are counted.
PLACEMENTS_PROGRAM = """
import json, os, re
import numpy as np
import lumen
import lumen.distributed as dist
import lumen.functional as F
from lumen.distributed import Partial, Replicate, Shard, from_local

n = int(os.environ["WORLD_SIZE"])
mesh = dist.init_device_mesh("cpu", (n,), mesh_dim_names=("tp",))
rank = mesh.get_rank()
host = lambda t: lumen.to_numpy(t).tolist()
mm = lambda x, w: F.matmul(x, w, "float32", "float32")
full = lumen.compile(lambda *ts: tuple(t.full_tensor() for t in ts))
out = {"rank": rank}

a = (np.arange(2 * n * n, dtype=np.float32) / (2 * n * n)).reshape(2 * n, n)
A = lumen.from_numpy(a)
PLACEMENTS = {"S0": Shard(0), "S1": Shard(1), "R": Replicate(), "P": Partial()}


def moved(x, dst):
    y = x.redistribute(placements=[dst])
    return y.to_local(), y.full_tensor()


out["local"] = {k: host(A.distribute(mesh, [p]).to_local()) for k, p in PLACEMENTS.items()}
out["pairs"] = {}
for s, src in PLACEMENTS.items():
    for t, dst in PLACEMENTS.items():
        local, whole = lumen.compile(moved)(A.distribute(mesh, [src]), dst)
        out["pairs"][f"{s}>{t}"] = [host(local), host(whole)]

x, y, r = (A.distribute(mesh, [p]) for p in (Shard(0), Shard(1), Replicate()))
p = from_local(lumen.from_numpy(a * (rank + 1)), mesh, [Partial()])
ops = lumen.compile(lambda x, y, r, p: (x + y, x * r, p + p, p * r, F.exp(p), x - p, p * 2.0, F.softmax(y, 1)))
mixed = ops(x, y, r, p)
out["mixed"] = [host(t) for t in full(*mixed)]
out["mixed_placements"] = [repr(t.placements) for t in mixed]
reduce = lumen.compile(lambda x, y, p: (F.sum(x, 0), F.sum(x, 1), F.sum(y, 0), F.amax(x, 0), F.sum(p, 1), F.mean(x)))
reduced = reduce(x, y, p)
out["reductions"] = [host(t) for t in full(*reduced)]
out["reduction_placements"] = [repr(t.placements) for t in reduced]

named = A.rename("seq", "hidden").distribute(mesh, [Shard("seq")])
s, t = lumen.compile(lambda x: (F.sum(x * 2.0, "hidden"), F.sum(x, "seq")))(named)
out["named"] = [list(s.names), repr(s.placements), list(t.names), repr(t.placements)]
out["named_values"] = [host(v.rename(None)) for v in full(s, t)]


class MLP(lumen.nn.Module):
    w1: lumen.Tensor
    w2: lumen.Tensor

    def __call__(self, x):
        return mm(F.tanh(mm(x, self.w1)), self.w2)


def step(model, x):
    loss = F.sum(F.tanh(model(x)))
    loss.backward()
    grads = [model.w1.grad, model.w2.grad]
    for w in (model.w1, model.w2):
        w.copy_(w - 0.1 * w.grad)
    return loss, grads


rng = np.random.default_rng(0)
xs, w1, w2 = (rng.standard_normal(s, np.float32) for s in [(4, 8), (8, 4 * n), (4 * n, 8)])
meshes = {"tp": (mesh, [Replicate()], [[Shard(1)], [Shard(0)]])}
if n == 4:
    grid = dist.DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"))
    meshes["dp_tp"] = (grid, [Shard(0), Replicate()], [[Replicate(), Shard(1)], [Replicate(), Shard(0)]])
for key, (m, data, (p1, p2)) in meshes.items():
    model = MLP(*(lumen.empty(list(w.shape), device="meta").distribute(m, p) for w, p in ((w1, p1), (w2, p2))))
    train = lumen.compile(step, device="cpu")
    train(model, lumen.empty([4, 8], device="meta").distribute(m, data))
    shards = (lumen.from_numpy(w).distribute(m, p).to_local() for w, p in ((w1, p1), (w2, p2)))
    model.load_state_dict(dict(zip(("w1", "w2"), shards)))
    loss, grads = train(model, lumen.from_numpy(xs).distribute(m, data))
    stepped = lumen.compile(lambda model: (model.w1.full_tensor(), model.w2.full_tensor()))(model)
    out[key] = {
        "loss": host(full(loss)[0]),
        "grads": [host(g) for g in full(*grads)],
        "grad_placements": [repr(g.placements) for g in grads],
        "stepped": [host(w) for w in stepped],
        "collectives": re.findall(r'bind\\("(all_\\w+|reduce_scatter)"', str(train)),
    }

x = lumen.empty([4, 8], device="meta")
W1 = lumen.empty([8, 4 * n], device="meta").distribute(mesh, [Shard(1)])
W2 = lumen.empty([4 * n, 8], device="meta").distribute(mesh, [Shard(0)])
mlp = lambda x, w1, w2: mm(F.tanh(mm(x, w1)), w2)
graphs = {
    "column": lambda x, w1: mm(x, w1),
    "row": lambda x, w2: F.tanh(mm(x, w2)),
    "mlp": lambda x, w1, w2: F.tanh(mlp(x, w1, w2)),
    "mlp_grad": lumen.grad(lambda x, w1, w2: F.sum(F.tanh(mlp(x, w1, w2))), (1, 2)),
}
h = lumen.empty([4, 4 * n], device="meta")
args = {"column": (x, W1), "row": (h, W2), "mlp": (x, W1, W2), "mlp_grad": (x, W1, W2)}
out["graphs"] = {}
for key, fn in graphs.items():
    text = str(lumen.make_graph(fn)(*args[key]))
    out["graphs"][key] = {c: text.count(c + "[") for c in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")}
print(json.dumps(out))
"""


def expected_model(x, w1, w2):
    """The loss of ``step``'s MLP, its weights' gradients and the weights after its SGD step, in float64."""
    x, w1, w2 = (v.astype(np.float64) for v in (x, w1, w2))
    h = np.tanh(x @ w1)
    y = h @ w2
    gy = 1 - np.tanh(y) ** 2
    g2 = h.T @ gy
    g1 = x.T @ (gy @ w2.T * (1 - h**2))
    return np.tanh(y).sum(), [g1, g2], [w1 - 0.1 * g1, w2 - 0.1 * g2]


@pytest.mark.parametrize("world_size", [2, 4])
def test_placements_across_processes_match_numpy(world_size):
    """Across CPU processes, every redistribution between Shard(0), Shard(1), Replicate and Partial keeps the
    global value; elementwise ops and reductions on mixed placements compute NumPy's values with the placements
    DTensor's strategies give (a sum over a sharded dim is partial); a tensor-parallel MLP module (and, on 4
    processes, one also data-parallel over a 2x2 mesh) gets NumPy's loss, gradients and SGD step, its gradients
    placed as its weights; and tensor parallelism traces exactly Megatron's collectives: none for a
    column-parallel matmul, one all-reduce for a row-parallel one before a nonlinearity, one more in its
    backward."""
    n = world_size
    results = launch(PLACEMENTS_PROGRAM, n)
    a = (np.arange(2 * n * n, dtype=np.float32) / (2 * n * n)).reshape(2 * n, n)
    total = n * (n + 1) / 2
    for rank, r in enumerate(results):
        locals_ = {
            "S0": np.split(a, n, 0)[rank],
            "S1": np.split(a, n, 1)[rank],
            "R": a,
            "P": a if rank == 0 else 0 * a,
        }
        for k, v in locals_.items():
            np.testing.assert_array_equal(r["local"][k], v)
        assert len(r["pairs"]) == 16
        for pair, (local, whole) in r["pairs"].items():
            np.testing.assert_allclose(local, locals_[pair.split(">")[1]], err_msg=pair)
            np.testing.assert_allclose(whole, a, err_msg=pair)

        expected = [a + a, a * a, 2 * total * a, total * a * a, np.exp(total * a), a - total * a, 2 * total * a]
        e = np.exp(a - a.max(1, keepdims=True))
        expected.append(e / e.sum(1, keepdims=True))
        for got, want in zip(r["mixed"], expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5)
        s0, s1, partial = "(Shard(dim=0),)", "(Shard(dim=1),)", "(Partial(reduce_op='sum'),)"
        assert r["mixed_placements"] == [s0, s0, partial, partial, "(Replicate(),)", s0, partial, s1]

        expected = [a.sum(0), a.sum(1), a.sum(0), a.max(0), total * a.sum(1), a.mean()]
        for got, want in zip(r["reductions"], expected, strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5)
        maximum = "(Partial(reduce_op='max'),)"
        assert r["reduction_placements"] == [partial, s0, s0, maximum, partial, partial]

        assert r["named"] == [["seq"], s0, ["hidden"], partial]
        np.testing.assert_allclose(r["named_values"][0], 2 * a.sum(1), rtol=1e-6)
        np.testing.assert_allclose(r["named_values"][1], a.sum(0), rtol=1e-6)

        rng = np.random.default_rng(0)
        xs, w1, w2 = (rng.standard_normal(s, np.float32) for s in [(4, 8), (8, 4 * n), (4 * n, 8)])
        loss, grads, stepped = expected_model(xs, w1, w2)
        meshes = {"tp": ["(Shard(dim=1),)", "(Shard(dim=0),)"]}
        if n == 4:
            meshes["dp_tp"] = ["(Replicate(), Shard(dim=1))", "(Replicate(), Shard(dim=0))"]
        for key, placements in meshes.items():
            got = r[key]
            np.testing.assert_allclose(got["loss"], loss, rtol=1e-5)
            for g, want in zip(got["grads"] + got["stepped"], grads + stepped, strict=True):
                np.testing.assert_allclose(g, want, rtol=1e-4, atol=1e-5)
            assert got["grad_placements"] == placements
        # Tensor parallelism: an all-reduce of the row-parallel output and one of its gradient; with data
        # parallelism too, each weight's gradient is summed over the dp processes.
        assert r["tp"]["collectives"] == ["all_reduce"] * 2
        if n == 4:
            assert r["dp_tp"]["collectives"] == ["all_reduce"] * 4

        none = {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0, "all_to_all": 0}
        assert r["graphs"] == {
            "column": none,
            "row": {**none, "all_reduce": 1},
            "mlp": {**none, "all_reduce": 1},
            "mlp_grad": {**none, "all_reduce": 2},
        }


def test_a_collective_takes_local_tensors():
    """A collective on a sharded tensor raises: it moves local values, which to_local() gives."""
    mesh = dist.init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    x = lumen.from_numpy(rand(4)).distribute(mesh, [Shard(0)])
    from lumen.graph import prims

    with pytest.raises(TypeError, match="to_local"):
        lumen.compile(lambda x: prims.all_reduce(x, "sum", [0]))(x)


# Run in each process: uneven shards (a dimension of 2n + 1) redistributed between every pair of placements and
# run through ops; on 4 processes, one dimension sharded over both dims of a 2x2 mesh; indexing, gathers,
# scatters and concatenation on sharded operands (no collectives); gradients through partial maxes and averages;
# and a module's sharded weights loaded from global tensors.
MORE_PROGRAM = """
import json, os, re
import numpy as np
import lumen
import lumen.distributed as dist
import lumen.functional as F
from lumen.distributed import Partial, Replicate, Shard, from_local
from lumen.graph import prims

n = int(os.environ["WORLD_SIZE"])
mesh = dist.init_device_mesh("cpu", (n,), mesh_dim_names=("tp",))
rank = mesh.get_rank()
host = lambda t: lumen.to_numpy(t).tolist()
full = lumen.compile(lambda *ts: tuple(t.full_tensor() for t in ts))
count = lambda graph: {c: str(graph).count(c + "[") for c in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")}
out = {"rank": rank}

# Uneven: 2n + 1 rows, n + 1 columns.
u = (np.arange((2 * n + 1) * (n + 1), dtype=np.float32) / 7).reshape(2 * n + 1, n + 1)
U = lumen.from_numpy(u)
PLACEMENTS = {"S0": Shard(0), "S1": Shard(1), "R": Replicate(), "P": Partial()}
out["uneven_local"] = {k: list(U.distribute(mesh, [p]).to_local().shape) for k, p in PLACEMENTS.items()}
moved = lumen.compile(lambda x, dst: (x.redistribute(placements=[dst]).to_local(), x.redistribute(placements=[dst]).full_tensor()))
out["uneven_pairs"] = {
    f"{s}>{t}": [list(local.shape), host(whole)]
    for s, src in PLACEMENTS.items()
    for t, dst in PLACEMENTS.items()
    for local, whole in [moved(U.distribute(mesh, [src]), dst)]
}
x, y = U.distribute(mesh, [Shard(0)]), U.distribute(mesh, [Shard(1)])
w = lumen.from_numpy(u.T.copy()).distribute(mesh, [Shard(0)])
ops = lumen.compile(lambda x, y, w: (F.exp(x) + x, F.sum(x, 0), F.sum(y, 0), F.amax(x, 0), F.matmul(y, w, "float32", "float32"), x * y))
res = ops(x, y, w)
out["uneven_ops"] = [host(t) for t in full(*res)]
out["uneven_placements"] = [repr(t.placements) for t in res]
back = from_local(U.distribute(mesh, [Shard(0)]).to_local(), mesh, [Shard(0)], shape=[2 * n + 1, n + 1])
out["from_local"] = [list(back.shape), host(full(back)[0])]

# Indexing on sharded operands: their strategies keep the shards (a lookup into rows sharded across processes
# gives a partial sum, DTensor's MaskPartial), so no collectives.
a = (np.arange(4 * n * 3, dtype=np.float32) / 5).reshape(4 * n, 3)
A = lumen.from_numpy(a)
rows = lumen.from_numpy(np.arange(2 * n, dtype=np.int32)[::-1].copy())

def indexing(a, rows):
    picked = a[rows]
    zero = (a * 0.0).to("int32")
    along = F.gather(a, 1, zero)
    joined = F.cat([a, a * 2.0], 1)
    block = prims.dynamic_slice(a, (0, 1), (a.shape[0], 2))
    added = prims.scatter_add(a, prims.iota("int32", (2,), 0), a[:, :2], 1)
    return picked, along, joined, block, added

inputs = (A.distribute(mesh, [Shard(0)]), rows)
indexed = lumen.compile(indexing)(*inputs)
out["indexing"] = [host(t) for t in full(*indexed)]
out["indexing_placements"] = [repr(t.placements) for t in indexed]
out["indexing_collectives"] = count(lumen.make_graph(indexing)(*inputs))

# Gradients through partial values: a max over a sharded dim (an all-reduce max in the program), and leaves
# partial by a max and by an average.
b = np.random.default_rng(1).standard_normal((2 * n, 3)).astype(np.float32)
B = lumen.from_numpy(b).distribute(mesh, [Shard(0)])
loss = lambda x: F.sum(F.tanh(F.amax(x, 0)))
g = lumen.compile(lumen.grad(loss))(B)
out["grad_max"] = [repr(g.placements), host(full(g)[0])]
part = lumen.from_numpy(b[:2] * (rank + 1))
leaf = lambda p: F.sum(F.tanh(p * 2.0))
for op in ("max", "avg"):
    p = from_local(part, mesh, [Partial(op)])
    gp = lumen.compile(lumen.grad(leaf))(p)
    out[f"leaf_{op}"] = [repr(gp.placements), host(gp.to_local()), host(full(gp)[0])]

# A module of sharded weights loaded from global tensors.
class Linear(lumen.nn.Module):
    w: lumen.Tensor

    def __call__(self, x):
        return F.matmul(x, self.w, "float32", "float32")

model = Linear(lumen.empty([3, 2 * n], device="meta").distribute(mesh, [Shard(1)]))
run = lumen.compile(lambda m, x: m(x).full_tensor(), device="cpu")
run(model, lumen.empty([4, 3], device="meta"))
model.load_state_dict({"w": lumen.from_numpy(b.T.copy())})
out["loaded"] = host(run(model, lumen.from_numpy(a[:4])))

# One tensor dimension sharded over both dims of a 2x2 mesh.
if n == 4:
    grid = dist.DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"))
    c = (np.arange(10 * 4, dtype=np.float32) / 3).reshape(10, 4)
    C = lumen.from_numpy(c)
    NESTED = {"S0S0": [Shard(0), Shard(0)], "S0S1": [Shard(0), Shard(1)], "RS0": [Replicate(), Shard(0)], "PR": [Partial(), Replicate()], "S1S0": [Shard(1), Shard(0)]}
    out["nested_local"] = {k: host(C.distribute(grid, p).to_local()) for k, p in NESTED.items()}
    out["nested_pairs"] = {
        f"{s}>{t}": [host(local), host(whole)]
        for s, src in NESTED.items()
        for t, dst in NESTED.items()
        for local, whole in [lumen.compile(lambda x: (x.redistribute(placements=dst).to_local(), x.redistribute(placements=dst).full_tensor()))(C.distribute(grid, src))]
    }
    z = C.distribute(grid, [Shard(0), Shard(0)])
    nested = lumen.compile(lambda z: (F.exp(z) * 2.0, F.sum(z, 0), F.sum(z, 1), F.matmul(z, z.transpose(0, 1), "float32", "float32")))(z)
    out["nested_ops"] = [host(t) for t in full(*nested)]
    out["nested_placements"] = [repr(t.placements) for t in nested]
print(json.dumps(out))
"""


def chunks(size, n):
    """``torch.chunk``'s blocks of a dimension of ``size`` split ``n`` ways: ``(start, length)`` each."""
    c = -(-size // n)
    return [(min(k * c, size), min(min(k * c, size) + c, size) - min(k * c, size)) for k in range(n)]


@pytest.mark.parametrize("world_size", [2, 4])
def test_more_placements_across_processes_match_numpy(world_size):
    """Across CPU processes: uneven shards follow torch.chunk (DTensor's) through every redistribution and
    through ops; from_local takes a global shape; gathers, scatters, slices and concatenation keep their shards
    with no collectives; gradients flow through partial maxes and averages; global tensors load into sharded
    weights; and, on a 2x2 mesh, a dimension sharded over both mesh dims redistributes to and from every
    placement and runs through ops."""
    n = world_size
    results = launch(MORE_PROGRAM, n)
    u = (np.arange((2 * n + 1) * (n + 1), dtype=np.float32) / 7).reshape(2 * n + 1, n + 1)
    s0, s1, partial = "(Shard(dim=0),)", "(Shard(dim=1),)", "(Partial(reduce_op='sum'),)"
    for rank, r in enumerate(results):
        rows, cols = chunks(2 * n + 1, n)[rank][1], chunks(n + 1, n)[rank][1]
        want = {"S0": [rows, n + 1], "S1": [2 * n + 1, cols], "R": [2 * n + 1, n + 1], "P": [2 * n + 1, n + 1]}
        assert r["uneven_local"] == want
        assert len(r["uneven_pairs"]) == 16
        for pair, (shape, whole) in r["uneven_pairs"].items():
            assert shape == want[pair.split(">")[1]], pair
            np.testing.assert_allclose(whole, u, rtol=1e-6, err_msg=pair)
        expected = [np.exp(u) + u, u.sum(0), u.sum(0), u.max(0), u @ u.T, u * u]
        for got, e in zip(r["uneven_ops"], expected, strict=True):
            np.testing.assert_allclose(got, e, rtol=1e-5)
        assert r["uneven_placements"] == [s0, partial, s0, "(Partial(reduce_op='max'),)", partial, s1]
        assert r["from_local"][0] == [2 * n + 1, n + 1]
        np.testing.assert_allclose(r["from_local"][1], u)

        a = (np.arange(4 * n * 3, dtype=np.float32) / 5).reshape(4 * n, 3)
        rows_ = np.arange(2 * n)[::-1]
        expected = [a[rows_], np.repeat(a[:, :1], 3, 1), np.concatenate([a, 2 * a], 1), a[:, 1:3], a + 0]
        expected[4][:, :2] *= 2
        for got, e in zip(r["indexing"], expected, strict=True):
            np.testing.assert_allclose(got, e, rtol=1e-6)
        assert r["indexing_placements"] == [partial] + [s0] * 4
        assert r["indexing_collectives"] == {"all_reduce": 0, "all_gather": 0, "reduce_scatter": 0, "all_to_all": 0}

        b = np.random.default_rng(1).standard_normal((2 * n, 3)).astype(np.float32)
        m = b.max(0)
        g = (b == m) * (1 - np.tanh(m) ** 2)
        assert r["grad_max"][0] == s0
        np.testing.assert_allclose(r["grad_max"][1], g, rtol=1e-5)
        parts = [b[:2] * (k + 1) for k in range(n)]
        for op, whole in (("max", np.max(parts, 0)), ("avg", np.mean(parts, 0))):
            placements, local, value = r[f"leaf_{op}"]
            assert placements == f"(Partial(reduce_op='{op}'),)"
            g = 2 * (1 - np.tanh(2 * whole) ** 2)
            np.testing.assert_allclose(local, g, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(value, g, rtol=1e-5, atol=1e-6)

        np.testing.assert_allclose(r["loaded"], a[:4] @ b.T, rtol=1e-5)

        if n == 4:
            c = (np.arange(10 * 4, dtype=np.float32) / 3).reshape(10, 4)
            i, j = divmod(rank, 2)
            first = chunks(10, 2)[i]
            inner = chunks(first[1], 2)[j]
            nested = c[first[0] + inner[0] : first[0] + inner[0] + inner[1]]
            col = chunks(4, 2)[j]
            local = {
                "S0S0": nested,
                "S0S1": c[slice(chunks(10, 2)[i][0], sum(chunks(10, 2)[i])), col[0] : col[0] + col[1]],
                "RS0": c[slice(chunks(10, 2)[j][0], sum(chunks(10, 2)[j]))],
                "PR": c if i == 0 else 0 * c,
                "S1S0": c[
                    slice(chunks(10, 2)[j][0], sum(chunks(10, 2)[j])),
                    slice(*[chunks(4, 2)[i][0], sum(chunks(4, 2)[i])]),
                ],
            }
            for k, v in local.items():
                np.testing.assert_allclose(r["nested_local"][k], v, err_msg=k)
            assert len(r["nested_pairs"]) == 25
            for pair, (got, whole) in r["nested_pairs"].items():
                np.testing.assert_allclose(got, local[pair.split(">")[1]], err_msg=pair)
                np.testing.assert_allclose(whole, c, err_msg=pair)
            expected = [np.exp(c) * 2, c.sum(0), c.sum(1), c @ c.T]
            for got, e in zip(r["nested_ops"], expected, strict=True):
                np.testing.assert_allclose(got, e, rtol=1e-5)
