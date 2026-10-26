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
        return all_reduce(t), all_reduce(t, "max"), all_gather(t, 1), reduce_scatter(t, 2)

    for out in lumen.compile(f)(X):
        np.testing.assert_array_equal(lumen.to_numpy(out.to("cpu")), x)
    labels = [s["label"] for s in lumen.graph_compiler.Plan(lumen.make_graph(f)(X), device).steps()]
    assert not any(c in label for label in labels for c in ("all_reduce", "all_gather", "reduce_scatter"))


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


# Run in each process on its own GPU: every collective over the world in three dtypes, their gradients, and,
# with 8 processes, all-reduces over a 2x4 mesh's slices.
COLLECTIVES_PROGRAM = """
import json, os
import numpy as np
import lumen
import lumen.functional as F
from lumen.distributed import all_gather, all_reduce, init_device_mesh, init_process_group, reduce_scatter

lumen.config.compiler.cuda.use_triton_backend = True
init_process_group()
rank, n = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
device = f"cuda:{rank}"
host = lambda t: lumen.to_numpy(t.to(dtype="float32").to("cpu")).tolist()
x = np.arange(12, dtype=np.float32).reshape(3, 4) + 100 * rank
y = np.arange(4 * n * n, dtype=np.float32).reshape(2 * n, 2 * n) + rank
out = {"rank": rank}
for dtype in ("float32", "bfloat16", "int32"):
    X, Y = (lumen.from_numpy(v).to(device).to(dtype=dtype) for v in (x, y))
    f = lambda X, Y: (all_reduce(X), all_reduce(X, "max"), all_reduce(X, "min"), all_gather(X, 0),
                      all_gather(X, 1), reduce_scatter(Y, 0), reduce_scatter(Y, 1))
    out[dtype] = [host(t) for t in lumen.compile(f)(X, Y)]
X, Y = lumen.from_numpy(x).to(device), lumen.from_numpy(y).to(device)
grads = lumen.compile(lumen.grad(lambda X, Y: F.sum(all_reduce(X) * (rank + 1))
                                 + F.sum(all_gather(X, 1) * (rank + 1))
                                 + F.sum(reduce_scatter(Y, 0) * (rank + 1)), (0, 1)))(X, Y)
out["grads"] = [host(g) for g in grads]
if n == 8:
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
    r = lumen.from_numpy(np.full((2,), float(rank), np.float32)).to(device)
    g = lambda r: (all_reduce(r, group=mesh["tp"]), all_reduce(r, group=mesh["dp"]), all_reduce(r, group=mesh))
    out["mesh"] = [host(t) for t in lumen.compile(g)(r)]
print(json.dumps(out))
"""


@pytest.mark.cuda
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_collectives_across_gpus_match_numpy(world_size):
    """A process per GPU reduces, gathers and scatters each other's tensors through NCCL, in float32, bfloat16
    and int32, along any dim; their gradients transpose into one another; with 8 processes, a 2x4 mesh's
    slices reduce over their own ranks."""
    gpus = "import lumen; print(lumen.cuda.device_count())"
    count = int(
        subprocess.run([sys.executable, "-c", gpus], capture_output=True, text=True, check=True).stdout.split()[-1]
    )
    if count < world_size:
        pytest.skip(f"needs {world_size} GPUs, found {count}")
    results = launch(COLLECTIVES_PROGRAM, world_size)
    n = world_size
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
            ]
            for g, e in zip(got, expected):
                np.testing.assert_allclose(g, e, rtol=rtol, atol=0, err_msg=dtype)
        scale = n * (n + 1) / 2
        gx, gy = (np.array(g) for g in r["grads"])
        np.testing.assert_array_equal(gx, np.full((3, 4), 2 * scale))
        rows = np.repeat(np.arange(1, n + 1, dtype=np.float32), 2)
        np.testing.assert_array_equal(gy, np.broadcast_to(rows[:, None], (2 * n, 2 * n)))
        if n == 8:
            dp, tp = divmod(rank, 4)
            assert r["mesh"] == [[float(sum(range(4 * dp, 4 * dp + 4)))] * 2, [float(tp + tp + 4)] * 2, [28.0] * 2]
