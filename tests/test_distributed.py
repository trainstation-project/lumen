import json
import os
import socket
import subprocess
import sys
from datetime import timedelta

import numpy as np
import pytest

import lumen
from lumen.distributed import DeviceMesh, TCPStore, get_rank, get_world_size, init_device_mesh


@pytest.fixture
def rank5_of_8(monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")


def test_rank_and_world_size_default_to_one_process(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert (get_rank(), get_world_size()) == (0, 1)


def test_rank_and_world_size_come_from_the_launcher(rank5_of_8):
    assert (get_rank(), get_world_size()) == (5, 8)


def test_init_device_mesh_is_row_major(rank5_of_8):
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
    assert mesh.mesh == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert (mesh.device_type, mesh.shape, mesh.ndim, mesh.size()) == ("cuda", (2, 4), 2, 8)
    assert mesh.mesh_dim_names == ("dp", "tp")
    assert (mesh.size(0), mesh.size("tp"), mesh.size(-1)) == (2, 4, 4)


def test_ranks(rank5_of_8):
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
    assert mesh.get_rank() == 5
    assert mesh.get_coordinate() == [1, 1]
    assert (mesh.get_local_rank("dp"), mesh.get_local_rank(1)) == (1, 1)
    with pytest.raises(RuntimeError, match="pass mesh_dim"):
        mesh.get_local_rank()


def test_slicing_keeps_the_submesh_of_the_rank(rank5_of_8):
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))
    assert mesh["dp"].mesh == [1, 5]
    assert mesh["tp"].mesh == [4, 5, 6, 7]
    assert mesh["tp"].get_local_rank() == 1
    assert mesh["tp", "dp"].mesh == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert mesh["dp", "tp"] == mesh
    with pytest.raises(KeyError, match="pp"):
        mesh["pp"]


def test_a_rank_outside_the_mesh(rank5_of_8):
    mesh = DeviceMesh("tpu", [[0, 1], [2, 3]], mesh_dim_names=("a", "b"))
    assert mesh.get_coordinate() is None
    with pytest.raises(RuntimeError, match="not in the mesh"):
        mesh.get_local_rank("a")
    with pytest.raises(KeyError, match="not in the mesh"):
        mesh["a"]


@pytest.mark.parametrize(
    "device_type, mesh, names, error",
    [
        ("gpu", [0, 1], None, "device_type"),
        ("cuda:0", [0, 1], None, "device_type"),
        ("cpu", [[0, 1], [2]], None, "rectangular"),
        ("cpu", [0, 0], None, "more than once"),
        ("cpu", [0, 8], None, "outside the world"),
        ("cpu", [-1, 0], None, "negative"),
        ("cpu", [[0, 1]], ("a",), "each of"),
        ("cpu", [[0, 1]], ("a", "a"), "unique"),
    ],
)
def test_invalid_meshes(rank5_of_8, device_type, mesh, names, error):
    with pytest.raises(ValueError, match=error):
        DeviceMesh(device_type, mesh, mesh_dim_names=names)


def test_equality_and_repr(rank5_of_8):
    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names=("dp", "tp"))
    assert mesh == init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
    assert hash(mesh) == hash(init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp")))
    assert mesh != init_device_mesh("cpu", (2, 2))
    assert repr(mesh) == "DeviceMesh('cpu', [[0, 1], [2, 3]], mesh_dim_names=('dp', 'tp'))"
    assert repr(init_device_mesh("cpu", (2,))) == "DeviceMesh('cpu', [0, 1])"


def test_a_compiled_function_takes_a_mesh(rank5_of_8):
    mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
    f = lumen.compile(lambda mesh, x: x * mesh.size("tp") + mesh["tp"].get_local_rank(), device="cpu")
    x = lumen.from_numpy(np.ones(3, np.float32))
    np.testing.assert_array_equal(lumen.to_numpy(f(mesh, x)), np.full(3, 5, np.float32))
    other = init_device_mesh("cpu", (4, 2), mesh_dim_names=("dp", "tp"))
    np.testing.assert_array_equal(lumen.to_numpy(f(other, x)), np.full(3, 3, np.float32))


def test_a_compiled_function_closes_over_a_mesh(rank5_of_8):
    mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp", "tp"))
    f = lumen.compile(lambda x: x + mesh.get_local_rank("dp"), device="cpu")
    x = lumen.from_numpy(np.zeros(3, np.float32))
    np.testing.assert_array_equal(lumen.to_numpy(f(x)), np.ones(3, np.float32))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
        assert lumen.distributed.is_initialized()
        assert (get_rank(), get_world_size()) == (0, 1)
        with pytest.raises(RuntimeError, match="already initialized"):
            lumen.distributed.init_process_group(store=store, rank=0, world_size=1)
    finally:
        lumen.distributed.destroy_process_group()
    assert not lumen.distributed.is_initialized()
    assert get_rank() == 3


def test_env_rendezvous_needs_the_launchers_variables(monkeypatch):
    for name in ("MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="MASTER_PORT is not set"):
        lumen.distributed.init_process_group(rank=0, world_size=1)
    with pytest.raises(ValueError, match="needs rank"):
        lumen.distributed.init_process_group(init_method="tcp://127.0.0.1:1", world_size=1)


RANK_PROGRAM = """
import json
import lumen
from lumen.distributed import get_rank, get_world_size, init_device_mesh, init_process_group

init_process_group(timeout=60)
rank = get_rank()
mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
print(json.dumps({"rank": rank, "world_size": get_world_size(), "tp": mesh["tp"].mesh, "coordinate": mesh.get_coordinate()}))
"""


def test_processes_meet_through_env_rendezvous():
    port = free_port()
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", RANK_PROGRAM],
            env={
                **os.environ,
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "RANK": str(rank),
                "WORLD_SIZE": "4",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in reversed(range(4))
    ]
    results = []
    for p in processes:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
        results.append(json.loads(out))
    results.sort(key=lambda r: r["rank"])
    assert [r["world_size"] for r in results] == [4] * 4
    assert [r["tp"] for r in results] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert [r["coordinate"] for r in results] == [[0, 0], [0, 1], [1, 0], [1, 1]]
