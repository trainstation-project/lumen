"""Process groups, device meshes and the TCP store, plus a helper that prepares single-host TPU launch variables."""

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

# chip count on one host -> LUMEN_TPU_TOPOLOGY host bounds
TPU_TOPOLOGIES = {1: "1,1,1", 4: "2,2,1", 8: "2,4,1"}


def _free_port():
    """Returns a TCP port the OS currently reports as free."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _tpu_chips():
    """Counts TPU chips from a fresh interpreter so this process never opens the TPU itself."""
    probe = "import lumen; print(lumen.tpu.device_count())"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
    return int(out.split()[-1])


def set_tpu_launch_env(nproc_per_node=None):
    """Sets the TPU topology, slicebuilder addresses and WORLD_SIZE env vars for `nproc_per_node` local workers.

    Counts local chips when `nproc_per_node` is None, keeps values already set, and returns the three variables.
    """
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
