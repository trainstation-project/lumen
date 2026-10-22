import json
import os
import socket
import subprocess
import sys
from datetime import timedelta

import pytest

import lumen
from lumen.distributed import DeviceMesh, TCPStore, get_rank, get_world_size, init_device_mesh, is_initialized

LAUNCHER = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")


@pytest.fixture(autouse=True)
def one_process(monkeypatch):
    for name in LAUNCHER:
        monkeypatch.delenv(name, raising=False)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(program, world_size, env=lambda rank: {}):
    port = free_port()
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
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
    assert (get_rank(), get_world_size()) == (0, 1)
    assert not is_initialized()


def test_a_mesh_of_one_process():
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
    with pytest.raises(ValueError, match=error):
        DeviceMesh(device_type, mesh, mesh_dim_names=names)


def test_equality_and_repr():
    mesh = DeviceMesh("cpu", [[0]], mesh_dim_names=("dp", "tp"))
    assert mesh == init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
    assert hash(mesh) == hash(init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp")))
    assert mesh != init_device_mesh("cpu", (1, 1))
    assert repr(mesh) == "DeviceMesh('cpu', [[0]], mesh_dim_names=('dp', 'tp'))"
    assert repr(init_device_mesh("cpu", (1,))) == "DeviceMesh('cpu', [0])"


@pytest.mark.mps
def test_an_mps_mesh_is_the_one_device_without_a_rendezvous():
    mesh = init_device_mesh("mps", (1,), mesh_dim_names=("dp",))
    assert (mesh.get_coordinate(), mesh.get_local_rank()) == ([0], 0)
    with pytest.raises(RuntimeError, match="the 1 MPS device, got 2"):
        init_device_mesh("mps", (2,))
    assert not is_initialized()


@pytest.mark.mps
def test_an_mps_mesh_runs_in_one_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="MPS runs in one process, but WORLD_SIZE is 2"):
        DeviceMesh("mps", [0])
    assert not is_initialized()


def test_a_mesh_of_several_processes_needs_the_launcher(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="MASTER_PORT is not set"):
        init_device_mesh("cpu", (2,))
    assert not is_initialized()


def test_store_set_get_add_wait():
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
    with pytest.raises(ValueError, match="MASTER_PORT is not set"):
        lumen.distributed.init_process_group(rank=0, world_size=1)
    with pytest.raises(ValueError, match="needs rank"):
        lumen.distributed.init_process_group(init_method="tcp://127.0.0.1:1", world_size=1)


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
    results = launch(INIT_PROGRAM, 4)
    assert [r["world_size"] for r in results] == [4] * 4
    assert [r["tp"] for r in results] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [r["coordinate"] for r in results] == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert [r["half"] for r in results] == [[0], [1], None, None]


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


TPU_PROGRAM = """
import json
import numpy as np
import lumen
from lumen.distributed import init_device_mesh

chips = lumen.tpu.device_count()
mesh = init_device_mesh("tpu", (2, 2), mesh_dim_names=("dp", "tp"))
f = lumen.compile(lambda x: x + mesh.get_rank(), device="tpu")
y = f(lumen.from_numpy(np.zeros((8, 128), np.float32)))
print(json.dumps({
    "rank": mesh.get_rank(),
    "chips": chips,
    "coordinate": mesh.get_coordinate(),
    "value": lumen.to_numpy(y.to("cpu")).tolist(),
}))
"""


def one_chip(rank):
    return {
        "TPU_VISIBLE_CHIPS": str(rank),
        "TPU_CHIPS_PER_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_PROCESS_PORT": str(8476 + rank),
    }


@pytest.mark.tpu
@pytest.mark.all_devices
def test_a_process_per_tpu_chip():
    count = "import lumen; print(lumen.tpu.device_count())"
    chips = int(subprocess.run([sys.executable, "-c", count], capture_output=True, text=True, check=True).stdout)
    if chips < 4:
        pytest.skip("needs 4 TPU chips")
    results = launch(TPU_PROGRAM, 4, one_chip)
    for rank, r in enumerate(results):
        assert r["chips"] == 1
        assert r["coordinate"] == list(divmod(rank, 2))
        assert r["value"] == [[float(rank)] * 128] * 8
