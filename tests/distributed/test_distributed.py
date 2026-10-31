"""Tests of lumen.distributed: device meshes, the TCP store, process groups, collectives and multi-process
launches."""

import json
import os
import socket
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pytest
from helpers import CUDA, MPS, TPU, meta

import lumen
import lumen.distributed as dist
from lumen.distributed import (
    DeviceMesh,
    TCPStore,
    all_gather,
    all_reduce,
    get_rank,
    get_world_size,
    init_device_mesh,
    is_initialized,
    reduce_scatter,
)
from lumen.graph import prims

# Environment variables a launcher like torchrun sets for each process.
LAUNCHER = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def one_process(monkeypatch):
    """Runs every test as a single process, without the launcher's variables."""
    for name in LAUNCHER:
        monkeypatch.delenv(name, raising=False)


def free_port():
    """A TCP port on localhost that is free right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(program, world_size, env=lambda rank: {}):
    """Runs `program` in `world_size` Python processes wired by env rendezvous; returns each one's last
    JSON line, sorted by rank. `env` adds variables per rank."""
    port = free_port()
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                **env(rank),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in reversed(range(world_size))
    ]
    results = []
    for p in processes:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err
        results.append(json.loads(out.splitlines()[-1]))
    return sorted(results, key=lambda r: r["rank"])


def test_rank_and_world_size_default_to_one_process():
    """Without a launcher, lumen sees rank 0 of a world of one, uninitialized."""
    assert (get_rank(), get_world_size()) == (0, 1)
    assert not is_initialized()


def test_a_mesh_of_one_process():
    """A 1x1 mesh needs no process group and answers its shape, names, rank and coordinate queries."""
    mesh = init_device_mesh("cuda", (1, 1), mesh_dim_names=("dp", "tp"))
    assert mesh.mesh == [[0]]
    assert (mesh.device_type, mesh.shape, mesh.ndim, mesh.size()) == ("cuda", (1, 1), 2, 1)
    assert mesh.mesh_dim_names == ("dp", "tp")
    assert (mesh.size(0), mesh.size("tp"), mesh.size(-1)) == (1, 1, 1)
    assert (mesh.get_rank(), mesh.get_coordinate()) == (0, [0, 0])
    with pytest.raises(RuntimeError, match="pass mesh_dim"):
        mesh.get_local_rank()
    with pytest.raises(KeyError, match="pp"):
        mesh["pp"]
    assert not is_initialized()


@pytest.mark.parametrize(
    "device_type, mesh, names, error",
    [
        ("gpu", [0], None, "device_type"),
        ("cuda:0", [0], None, "device_type"),
        ("cpu", [[0, 1], [2]], None, "rectangular"),
        ("cpu", [0, 0], None, "more than once"),
        ("cpu", [0, 1], None, "outside the world"),
        ("cpu", [-1], None, "negative"),
        ("cpu", [[0]], ("a",), "each of"),
        ("cpu", [[0]], ("a", "a"), "unique"),
    ],
)
def test_invalid_meshes(device_type, mesh, names, error):
    """Malformed meshes raise naming what is wrong."""
    with pytest.raises(ValueError, match=error):
        DeviceMesh(device_type, mesh, mesh_dim_names=names)


def test_equality_and_repr():
    """Meshes compare and hash by device type, layout and names, and repr as their constructor call."""
    mesh = DeviceMesh("cpu", [[0]], mesh_dim_names=("dp", "tp"))
    assert mesh == init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    assert hash(mesh) == hash(init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp")))
    assert mesh != init_device_mesh("cpu", (1, 1))
    assert repr(mesh) == "DeviceMesh('cpu', [[0]], mesh_dim_names=('dp', 'tp'))"
    assert repr(init_device_mesh("cpu", (1,))) == "DeviceMesh('cpu', [0])"


@pytest.mark.mps
def test_an_mps_mesh_is_the_one_device_without_a_rendezvous():
    """An MPS mesh is the single local device and starts no process group."""
    mesh = init_device_mesh("mps", (1,), mesh_dim_names=("dp",))
    assert (mesh.get_coordinate(), mesh.get_local_rank()) == ([0], 0)
    with pytest.raises(RuntimeError, match="the 1 MPS device, got 2"):
        init_device_mesh("mps", (2,))
    assert not is_initialized()


@pytest.mark.mps
def test_an_mps_mesh_runs_in_one_process(monkeypatch):
    """An MPS mesh rejects a world of several processes."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="MPS runs in one process, but WORLD_SIZE is 2"):
        DeviceMesh("mps", [0])
    assert not is_initialized()


def test_a_mesh_of_several_processes_needs_the_launcher(monkeypatch):
    """A multi-process mesh outside a launcher reports the missing rendezvous variable."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="MASTER_PORT is not set"):
        init_device_mesh("cpu", (2,))
    assert not is_initialized()


def test_store_set_get_add_wait():
    """A master and a client share keys: set, get, add, and wait with a timeout."""
    master = TCPStore("127.0.0.1", 0, is_master=True, timeout=timedelta(seconds=10))
    client = TCPStore("127.0.0.1", master.port, timeout=10)
    master.set("text", "hello")
    client.set("raw", b"\x00\x01")
    assert (client.get("text"), master.get("raw")) == (b"hello", b"\x00\x01")
    assert (master.add("n", 3), client.add("n", 4), client.get("n")) == (3, 7, b"7")
    client.wait(["text", "n"])
    with pytest.raises(RuntimeError, match="timed out"):
        client.wait(["missing"], timeout=0.05)


def test_a_world_of_one_process(monkeypatch):
    """A process group of one initializes once, and destroying it restores the env rank."""
    monkeypatch.setenv("RANK", "3")
    store = TCPStore("127.0.0.1", 0, is_master=True, timeout=10)
    lumen.distributed.init_process_group(store=store, rank=0, world_size=1)
    try:
        assert is_initialized()
        assert (get_rank(), get_world_size()) == (0, 1)
        with pytest.raises(RuntimeError, match="already initialized"):
            lumen.distributed.init_process_group(store=store, rank=0, world_size=1)
    finally:
        lumen.distributed.destroy_process_group()
    assert not is_initialized()
    assert get_rank() == 3


def test_env_rendezvous_needs_the_launchers_variables():
    """Env rendezvous without the launcher's variables, or a TCP init without a rank, fails clearly."""
    with pytest.raises(ValueError, match="MASTER_PORT is not set"):
        lumen.distributed.init_process_group(rank=0, world_size=1)
    with pytest.raises(ValueError, match="needs rank"):
        lumen.distributed.init_process_group(init_method="tcp://127.0.0.1:1", world_size=1)


# Run in each process: rendezvous, build a 2x2 mesh and report what this rank sees.
INIT_PROGRAM = """
import json
from lumen.distributed import DeviceMesh, get_rank, get_world_size, init_device_mesh, init_process_group

init_process_group(timeout=60)
mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
print(json.dumps({
    "rank": get_rank(),
    "world_size": get_world_size(),
    "tp": mesh["tp"].mesh,
    "coordinate": mesh.get_coordinate(),
    "half": DeviceMesh("cpu", [0, 1]).get_coordinate(),
}))
"""


def test_processes_meet_through_env_rendezvous():
    """Four processes rendezvous and each sees its row of a 2x2 mesh."""
    results = launch(INIT_PROGRAM, 4)
    assert [r["world_size"] for r in results] == [4] * 4
    assert [r["tp"] for r in results] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [r["coordinate"] for r in results] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert [r["half"] for r in results] == [[0], [1], None, None]


# Run in each process: a 2x4 mesh that initializes the group itself and is read by compiled functions.
MESH_PROGRAM = """
import json
import numpy as np
import lumen
from lumen.distributed import init_device_mesh, is_initialized

mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
f = lumen.compile(lambda mesh, x: x * mesh.size("tp") + mesh["tp"].get_local_rank(), device="cpu")
g = lumen.compile(lambda x: x + mesh.get_local_rank("dp"), device="cpu")
x = lumen.from_numpy(np.ones(1, np.float32))
print(json.dumps({
    "rank": mesh.get_rank(),
    "initialized": is_initialized(),
    "mesh": mesh.mesh,
    "coordinate": mesh.get_coordinate(),
    "local": [mesh.get_local_rank("dp"), mesh.get_local_rank(1)],
    "dp": mesh["dp"].mesh,
    "tp": mesh["tp"].mesh,
    "tp_dp": mesh["tp", "dp"].mesh,
    "whole": mesh["dp", "tp"] == mesh,
    "f": float(lumen.to_numpy(f(mesh, x))[0]),
    "g": float(lumen.to_numpy(g(x))[0]),
}))
"""


def test_a_mesh_meets_its_processes_and_compiles_with_them():
    """Eight processes build a 2x4 mesh, slice it by name, and compiled functions read their mesh ranks."""
    results = launch(MESH_PROGRAM, 8)
    for rank, r in enumerate(results):
        dp, tp = divmod(rank, 4)
        assert r["initialized"] and r["whole"]
        assert r["mesh"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert r["coordinate"] == r["local"] == [dp, tp]
        assert r["dp"] == [tp, tp + 4]
        assert r["tp"] == [4 * dp + i for i in range(4)]
        assert r["tp_dp"] == [[0, 4], [1, 5], [2, 6], [3, 7]]
        assert (r["f"], r["g"]) == (4 + tp, 1 + dp)


# Run in each process: a 2x2 TPU mesh with one chip per process, adding the rank on device.
TPU_PROGRAM = """
import json
import numpy as np
import lumen
from lumen.distributed import init_device_mesh

mesh = init_device_mesh("tpu", (2, 2), mesh_dim_names=("dp", "tp"))
chips = lumen.tpu.device_count()
f = lumen.compile(lambda x: x + mesh.get_rank(), device="tpu")
y = f(lumen.from_numpy(np.zeros((8, 128), np.float32)))
print(json.dumps({
    "rank": mesh.get_rank(),
    "chips": chips,
    "coordinate": mesh.get_coordinate(),
    "value": lumen.to_numpy(y.to("cpu")).tolist(),
}))
"""


def test_tpu_launch_env(monkeypatch):
    """`set_tpu_launch_env` derives the topology and slice-builder addresses for a process per chip."""
    monkeypatch.setattr(os, "environ", {"LUMEN_TPU_SLICEBUILDER_ADDRESSES": "h:1,h:2,h:3,h:4"})
    env = lumen.distributed.set_tpu_launch_env(nproc_per_node=4)
    assert env == {
        "LUMEN_TPU_TOPOLOGY": "2,2,1",
        "LUMEN_TPU_SLICEBUILDER_ADDRESSES": "h:1,h:2,h:3,h:4",
        "WORLD_SIZE": "4",
    }
    monkeypatch.setattr(os, "environ", {})
    addresses = lumen.distributed.set_tpu_launch_env(nproc_per_node=8)["LUMEN_TPU_SLICEBUILDER_ADDRESSES"].split(",")
    assert len(set(addresses)) == 8 and all(a.startswith("localhost:") for a in addresses)
    monkeypatch.setattr(os, "environ", {})
    with pytest.raises(ValueError, match="set LUMEN_TPU_TOPOLOGY"):
        lumen.distributed.set_tpu_launch_env(nproc_per_node=3)


def tpu_launch_env():
    """Variables `python -m lumen.distributed` prints for launching a process per TPU chip."""
    out = subprocess.run(
        [sys.executable, "-m", "lumen.distributed"], capture_output=True, text=True, check=True
    ).stdout
    return dict(line.replace('"', "").split("=", 1) for line in out.splitlines())


@pytest.mark.tpu
@pytest.mark.all_devices
def test_a_process_per_tpu_chip_in_one_slice():
    """On a 4-chip host, a process per chip sees one chip and its coordinate in a 2x2 mesh."""
    count = "import lumen; print(lumen.tpu.device_count())"
    chips = int(subprocess.run([sys.executable, "-c", count], capture_output=True, text=True, check=True).stdout)
    if chips != 4:
        pytest.skip("needs a host of 4 TPU chips")
    env = tpu_launch_env()
    assert env["LUMEN_TPU_TOPOLOGY"] == "2,2,1" and env["WORLD_SIZE"] == "4"
    results = launch(TPU_PROGRAM, 4, lambda rank: env)
    for rank, r in enumerate(results):
        assert r["chips"] == 1
        assert r["coordinate"] == list(divmod(rank, 2))
        assert r["value"] == [[float(rank)] * 128] * 8


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA, TPU])
def test_collectives_of_one_process_are_its_values(device, triton_on_cuda):
    """In a world of one process, every collective is its operand (gathered and scattered along any dim) and
    compiles to no collective step, on any device."""
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    try:
        X = lumen.from_numpy(x).to(device)
    except RuntimeError as e:
        pytest.skip(str(e))

    def f(t):
        """Every collective of ``t``, over the whole (one-process) world."""
        return (
            all_reduce(t),
            all_reduce(t, "max"),
            all_gather(t, 1),
            reduce_scatter(t, 2),
            dist.all_to_all(t, 1, 2),
            dist.broadcast(t),
            dist.reduce(t),
            dist.gather(t, dim=1),
            dist.scatter(t, dim=2),
        )

    for out in lumen.compile(f)(X):
        np.testing.assert_array_equal(lumen.to_numpy(out.to("cpu")), x)
    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(X), device).steps()]
    names = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "reduce", "gather", "scatter")
    assert not any(c in label for label in labels for c in names)


def test_collective_primitives_type_and_print():
    """A collective records its group's ranks: ``all_gather`` stacks a leading axis of one row per process,
    ``reduce_scatter`` takes one off, and the graph prints the op and ranks."""
    graph = lumen.make_graph(
        lambda x, y: (
            prims.all_reduce(x, "max", [0, 2]),
            prims.all_gather(x, [0, 2]),
            prims.reduce_scatter(y, "sum", [0, 2]),
        )
    )(meta(3, 4), meta(2, 5))
    text = str(graph)
    assert "all_reduce[op=max ranks=(0, 2)]" in text and "all_gather[ranks=(0, 2)]" in text
    assert "reduce_scatter[op=sum ranks=(0, 2)]" in text
    assert "f32[2,3,4] = all_gather" in text and "f32[5] = reduce_scatter" in text
    with pytest.raises(ValueError, match="one row for each of the 2 processes"):
        lumen.make_graph(lambda y: prims.reduce_scatter(y, "sum", [0, 2]))(meta(3, 5))


def test_rooted_and_point_to_point_primitives_type_and_print():
    """``all_to_all`` keeps its operand's type (a row per process), ``gather_to`` stacks a leading axis and
    ``scatter_from`` takes one off, ``broadcast`` and ``reduce`` keep the type, a ``send`` returns an empty
    token and a ``recv`` its own dtype and shape; each prints its root or peer, which must be in its group."""
    graph = lumen.make_graph(
        lambda x, y: (
            prims.all_to_all(y, [0, 2]),
            prims.broadcast(x, 2, [0, 2]),
            prims.reduce(x, "min", 0, [0, 2]),
            prims.gather_to(x, 2, [0, 2]),
            prims.scatter_from(y, 0, [0, 2]),
            prims.send(x, 2, [0, 2]),
            prims.recv([7], "int32", 2, [0, 2]),
        )
    )(meta(3, 4), meta(2, 5))
    text = str(graph)
    assert "f32[2,5] = all_to_all[ranks=(0, 2)]" in text
    assert "f32[3,4] = broadcast[ranks=(0, 2) root=2]" in text
    assert "f32[3,4] = reduce[op=min ranks=(0, 2) root=0]" in text
    assert "f32[2,3,4] = gather_to[ranks=(0, 2) root=2]" in text
    assert "f32[5] = scatter_from[ranks=(0, 2) root=0]" in text
    assert "f32[0] = send[peer=2 ranks=(0, 2)]" in text
    assert "i32[7] = recv[dtype=i32 peer=2 ranks=(0, 2) shape=(7,)]" in text
    with pytest.raises(ValueError, match="root 1 is not in the group"):
        lumen.make_graph(lambda x: prims.broadcast(x, 1, [0, 2]))(meta(3))
    with pytest.raises(ValueError, match="peer 3 is not in the group"):
        lumen.make_graph(lambda x: prims.send(x, 3, [0, 2]))(meta(3))
    with pytest.raises(ValueError, match="one row for each of the 2 processes"):
        lumen.make_graph(lambda y: prims.all_to_all(y, [0, 2]))(meta(3, 5))


def test_a_send_runs_though_nothing_reads_it():
    """``send`` returns nothing, yet the trace keeps it: its token is an output after the returned ones, and
    two equal sends (or receives) stay two messages."""

    def f(x):
        """Sends ``x`` twice and receives two tensors like it, returning their sum."""
        dist.send(x, 1, group=[0, 1])
        dist.send(x, 1, group=[0, 1])
        return dist.recv(x, 1, group=[0, 1]) + dist.recv(x, 1, group=[0, 1])

    graph = lumen.make_graph(f)(meta(4))
    text = str(graph)
    assert text.count("= send[") == 2 and text.count("= recv[") == 2
    assert len(graph.outputs()) == 3
    labels = [s["label"] for s in lumen.graph_compiler.Plan(graph, "cpu").steps()]
    assert [label for label in labels if label in ("send", "recv")] == ["send", "send", "recv", "recv"]


@pytest.mark.parametrize("device", ["cpu", MPS, CUDA])
def test_collectives_overlap_the_work_independent_of_them(device):
    """``lumen.distributed``'s collectives end in a ``device_wait``. The plan starts the collective as soon as
    its operand exists and waits for it just before the first op that reads the result, with the work
    independent of it in between. Until then it keeps the operand's buffer (which NCCL reads) to itself."""
    import lumen.functional as F

    def f(x, y):
        """An all-reduce of ``2 x`` plus one, and ``sum(3 exp y)``, which doesn't need it."""
        return prims.device_wait(prims.all_reduce(x * 2.0, "sum", [0, 1])) + 1.0, F.sum(F.exp(y) * 3.0)

    steps = lumen.graph_compiler.Plan(lumen.make_graph(f)(meta(4), meta(4)), device).steps()
    labels = [s["label"] for s in steps]
    start, wait = labels.index("all_reduce"), labels.index("device_wait")
    assert start < next(i for i, label in enumerate(labels) if label.startswith("exp")) < wait
    assert labels[wait + 1] == "add" and steps[wait]["inputs"][0] == steps[wait]["output"]
    operand = steps[start]["inputs"][0][0]
    assert all(s["output"][0] != operand for s in steps[start + 1 : wait])
    graph = str(lumen.make_graph(lambda x: all_reduce(x, group=[0, 1]))(meta(4)))
    assert "all_reduce[op=sum ranks=(0, 1)]" in graph and "device_wait" in graph


# Run in each process (on its own GPU for CUDA): every collective over the world in three dtypes, their
# gradients, a ring of sends and receives, a barrier, and, with 8 processes, collectives over a 2x4 mesh's slices.
COLLECTIVES_PROGRAM = """
import json, os, time
import numpy as np
import lumen
import lumen.distributed as dist
import lumen.functional as F

lumen.config.compiler.cuda.use_triton_backend = True
dist.init_process_group()
rank, n = dist.get_rank(), dist.get_world_size()
device = f"cuda:{rank}" if os.environ["LUMEN_TEST_DEVICE"] == "cuda" else "cpu"
host = lambda t: lumen.to_numpy(t.to(dtype="float32").to("cpu")).tolist()
x = np.arange(12, dtype=np.float32).reshape(3, 4) + 100 * rank
y = np.arange(4 * n * n, dtype=np.float32).reshape(2 * n, 2 * n) + rank
last, c = n - 1, rank + 1
out = {"rank": rank}
for dtype in ("float32", "bfloat16", "int32"):
    X, Y = (lumen.from_numpy(v).to(device).to(dtype=dtype) for v in (x, y))
    f = lambda X, Y: (dist.all_reduce(X), dist.all_reduce(X, "max"), dist.all_reduce(X, "min"),
                      dist.all_gather(X, 0), dist.all_gather(X, 1), dist.reduce_scatter(Y, 0),
                      dist.reduce_scatter(Y, 1), dist.all_to_all(Y, 0, 1), dist.all_to_all(Y, 1, 0),
                      dist.broadcast(X, last), dist.reduce(X, last), dist.reduce(X, 0, "max"),
                      dist.gather(X, last, 1), dist.scatter(Y, last, 1))
    out[dtype] = [host(t) for t in lumen.compile(f)(X, Y)]
X, Y = lumen.from_numpy(x).to(device), lumen.from_numpy(y).to(device)
grads = lumen.compile(lumen.grad(lambda X, Y: F.sum(dist.all_reduce(X) * c) + F.sum(dist.all_gather(X, 1) * c)
                                 + F.sum(dist.reduce_scatter(Y, 0) * c), (0, 1)))(X, Y)
out["grads"] = [host(g) for g in grads]
grads = lumen.compile(lumen.grad(lambda X, Y: F.sum(dist.all_to_all(Y, 0, 1) * c) + F.sum(dist.broadcast(X, last) * c)
                                 + F.sum(dist.reduce(X, 0) * c) + F.sum(dist.gather(X, last, 1) * c)
                                 + F.sum(dist.scatter(Y, last, 1) * c), (0, 1)))(X, Y)
out["rooted_grads"] = [host(g) for g in grads]

def ring(X):
    \"\"\"Sends X, X and 3 X to the next rank and receives the previous rank's, even ranks sending first.\"\"\"
    sends = lambda: [dist.send(t, (rank + 1) % n) for t in (X, X, X * 3.0)]
    recvs = lambda: [dist.recv(X, (rank - 1) % n) for _ in range(3)]
    if rank % 2 == 0:
        sends()
        return recvs()
    got = recvs()
    sends()
    return got

out["ring"] = [host(t) for t in lumen.compile(ring)(X)]
# Rank 0 arrives late at the barrier: every rank leaves it after rank 0 wrote its file.
mark = os.path.join(os.environ["LUMEN_TEST_DIR"], "arrived")
if rank == 0:
    time.sleep(0.5)
    open(mark, "w").close()
dist.barrier()
out["barrier"] = os.path.exists(mark)
dist.barrier(list(range(n)))
if n == 8:
    mesh = dist.init_device_mesh(device.split(":")[0], (2, 4), mesh_dim_names=("dp", "tp"))
    r = lumen.from_numpy(np.full((2,), float(rank), np.float32)).to(device)
    g = lambda r: (dist.all_reduce(r, group=mesh["tp"]), dist.all_reduce(r, group=mesh["dp"]),
                   dist.all_reduce(r, group=mesh), dist.broadcast(r, mesh["tp"].mesh[0], group=mesh["tp"]))
    out["mesh"] = [host(t) for t in lumen.compile(g)(r)]
print(json.dumps(out))
"""


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_collectives_across_processes_match_numpy(device, world_size, tmp_path):
    """A process per device (each GPU, or CPU processes over the store) reduces, gathers, scatters, exchanges
    and broadcasts each other's tensors, in float32, bfloat16 and int32, along any dim; their gradients
    transpose into one another; sends and receives pair in program order around a ring; a barrier holds
    every process until the last arrives; with 8 processes, a 2x4 mesh's slices reduce over their own ranks."""
    if device == "cuda":
        gpus = "import lumen; print(lumen.cuda.device_count())"
        out = subprocess.run([sys.executable, "-c", gpus], capture_output=True, text=True, check=True).stdout
        if int(out.split()[-1]) < world_size:
            pytest.skip(f"needs {world_size} GPUs, found {out.split()[-1]}")
    env = {"LUMEN_TEST_DEVICE": device, "LUMEN_TEST_DIR": str(tmp_path)}
    results = launch(COLLECTIVES_PROGRAM, world_size, lambda rank: env)
    n, last = world_size, world_size - 1
    xs = [np.arange(12, dtype=np.float32).reshape(3, 4) + 100 * r for r in range(n)]
    ys = [np.arange(4 * n * n, dtype=np.float32).reshape(2 * n, 2 * n) + r for r in range(n)]
    total = sum(ys)
    for rank, r in enumerate(results):
        for dtype in ("float32", "bfloat16", "int32"):
            got = [np.array(v) for v in r[dtype]]
            # bfloat16 sums round once per rank's addition: n roundings of 2^-8 each.
            rtol = n * 2**-8 if dtype == "bfloat16" else 0
            expected = [
                sum(xs),
                np.max(xs, 0),
                np.min(xs, 0),
                np.concatenate(xs, 0),
                np.concatenate(xs, 1),
                np.split(total, n, 0)[rank],
                np.split(total, n, 1)[rank],
                np.concatenate([np.split(ys[j], n, 0)[rank] for j in range(n)], 1),
                np.concatenate([np.split(ys[j], n, 1)[rank] for j in range(n)], 0),
                xs[last],
                sum(xs) if rank == last else np.zeros((3, 4)),
                np.max(xs, 0) if rank == 0 else np.zeros((3, 4)),
                np.concatenate(xs, 1) if rank == last else np.zeros((3, 4 * n)),
                np.split(ys[last], n, 1)[rank],
            ]
            for g, e in zip(got, expected, strict=True):
                np.testing.assert_allclose(g, e, rtol=rtol, atol=0, err_msg=dtype)
        scale = n * (n + 1) / 2
        gx, gy = (np.array(g) for g in r["grads"])
        np.testing.assert_array_equal(gx, np.full((3, 4), 2 * scale))
        rows = np.repeat(np.arange(1, n + 1, dtype=np.float32), 2)
        np.testing.assert_array_equal(gy, np.broadcast_to(rows[:, None], (2 * n, 2 * n)))
        # broadcast: the root's value reached every rank; reduce: rank 0 weighs every rank's by 1; gather:
        # the last rank weighs every rank's by n; all_to_all: rank j's row block i reached rank i; scatter:
        # the last rank's column block j reached rank j.
        gx, gy = (np.array(g) for g in r["rooted_grads"])
        np.testing.assert_array_equal(gx, np.full((3, 4), (scale if rank == last else 0) + 1 + n))
        columns = rows[None, :] if rank == last else 0
        np.testing.assert_array_equal(gy, rows[:, None] + columns + np.zeros((2 * n, 2 * n)))
        previous = xs[(rank - 1) % n]
        assert r["ring"] == [previous.tolist(), previous.tolist(), (3 * previous).tolist()]
        assert r["barrier"]
        if n == 8:
            dp, tp = divmod(rank, 4)
            assert r["mesh"] == [
                [float(sum(range(4 * dp, 4 * dp + 4)))] * 2,
                [float(tp + tp + 4)] * 2,
                [28.0] * 2,
                [float(4 * dp)] * 2,
            ]
