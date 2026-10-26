"""Process groups, device meshes, the TCP store and collectives (NCCL on CUDA), plus a helper that prepares
single-host TPU launch variables. Each collective runs asynchronously and ends in a ``device_wait``, which
the plan places just before the first op that reads the result."""

import ctypes
import importlib.util
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
    "all_gather",
    "all_reduce",
    "destroy_process_group",
    "get_rank",
    "get_world_size",
    "init_device_mesh",
    "init_process_group",
    "is_initialized",
    "reduce_scatter",
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


def _load_nccl():
    """Loads ``libnccl.so.2`` from the nvidia-nccl-cu12/-cu13 wheel, as ``torch/__init__.py`` does, so lumen's
    CUDA collectives find it; does nothing where the wheel isn't installed."""
    if sys.platform != "linux":
        return
    spec = importlib.util.find_spec("nvidia.nccl") if importlib.util.find_spec("nvidia") else None
    for root in spec.submodule_search_locations if spec else []:
        path = os.path.join(root, "lib", "libnccl.so.2")
        if os.path.exists(path):
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            return


_load_nccl()


def _ranks(group):
    """The global ranks a collective spans: every process for ``None``, a mesh's ranks for a ``DeviceMesh`` (one
    slice, ``mesh["tp"]``, spans this process's peers along it), or the given ranks."""
    if group is None:
        return list(range(get_world_size()))
    if isinstance(group, DeviceMesh):
        ranks, todo = [], [group.mesh]
        while todo:
            item = todo.pop(0)
            if isinstance(item, list):
                todo[:0] = item
            else:
                ranks.append(item)
        return ranks
    return [int(r) for r in group]


def all_reduce(tensor, op="sum", group=None):
    """``tensor`` combined across ``group``'s processes with ``op`` (``"sum"``, ``"max"`` or ``"min"``); every one
    of them gets the result. Traced inside ``lumen.compile`` (PyTorch's functional ``all_reduce``)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    return tensor if len(ranks) == 1 else prims.device_wait(prims.all_reduce(tensor, op, ranks))


def all_gather(tensor, dim=0, group=None):
    """Every process in ``group``'s ``tensor`` concatenated along ``dim``, in rank order (PyTorch's
    ``all_gather_into_tensor``, any ``dim``)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    dim = dim % tensor.ndim
    gathered = prims.device_wait(prims.all_gather(tensor, ranks))
    # The gathered axis goes just before `dim`, then merges with it: rank r's rows come r-th along `dim`.
    order = [*range(1, dim + 1), 0, *range(dim + 1, tensor.ndim + 1)]
    moved = prims.transpose(gathered, order) if dim else gathered
    shape = [*tensor.shape[:dim], len(ranks) * tensor.shape[dim], *tensor.shape[dim + 1 :]]
    return prims.reshape(moved, shape)


def reduce_scatter(tensor, dim=0, op="sum", group=None):
    """``tensor`` combined across ``group``'s processes with ``op``, each keeping its share of ``dim`` (split
    into one equal block per process, in rank order): PyTorch's ``reduce_scatter_tensor``, any ``dim``."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    dim, n = dim % tensor.ndim, len(ranks)
    if tensor.shape[dim] % n:
        raise ValueError(
            f"reduce_scatter(): dimension {dim} of size {tensor.shape[dim]} doesn't split over {n} processes"
        )
    # Split `dim` into (process, block) and move the process axis first, where the primitive scatters it.
    split = prims.reshape(tensor, [*tensor.shape[:dim], n, tensor.shape[dim] // n, *tensor.shape[dim + 1 :]])
    order = [dim, *range(dim), *range(dim + 1, tensor.ndim + 1)]
    rows = prims.transpose(split, order) if dim else split
    return prims.device_wait(prims.reduce_scatter(rows, op, ranks))
