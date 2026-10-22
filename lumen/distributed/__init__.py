import os
import socket
import subprocess
import sys

from lumen.distributed._C import (
    DeviceMesh,
    TCPStore,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_device_mesh,
    init_process_group,
    is_initialized,
)

__all__ = [
    "DeviceMesh",
    "TCPStore",
    "destroy_process_group",
    "get_rank",
    "get_world_size",
    "init_device_mesh",
    "init_process_group",
    "is_initialized",
    "set_tpu_launch_env",
]

TPU_TOPOLOGIES = {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"}


def _free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _tpu_chips():
    probe = "import lumen; print(lumen.tpu.device_count())"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    return int(out.split()[-1])


def set_tpu_launch_env(nproc_per_node=None):
    count = nproc_per_node or _tpu_chips()
    if not count:
        raise ValueError("set_tpu_launch_env(): expected TPU devices, got none")
    if "LUMEN_TPU_TOPOLOGY" not in os.environ:
        if count not in TPU_TOPOLOGIES:
            raise ValueError(
                f"set_tpu_launch_env(): no known single-host topology of {count} TPU chips; set LUMEN_TPU_TOPOLOGY"
            )
        os.environ["LUMEN_TPU_TOPOLOGY"] = TPU_TOPOLOGIES[count]
    os.environ.setdefault(
        "LUMEN_TPU_SLICEBUILDER_ADDRESSES", ",".join(f"localhost:{_free_port()}" for _ in range(count))
    )
    os.environ.setdefault("WORLD_SIZE", str(count))
    return {
        name: os.environ[name] for name in ("LUMEN_TPU_TOPOLOGY", "LUMEN_TPU_SLICEBUILDER_ADDRESSES", "WORLD_SIZE")
    }
