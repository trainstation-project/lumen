"""Tests of lumen.distributed: device meshes, the TCP store, process groups and multi-process launches."""

import json
import os
import socket
import subprocess
import sys
from datetime import timedelta

import pytest

import lumen
from lumen.distributed import DeviceMesh, TCPStore, get_rank, get_world_size, init_device_mesh, is_initialized

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
