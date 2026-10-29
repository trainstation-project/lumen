"""Process groups, device meshes, the TCP store, barriers and collectives (NCCL on CUDA, the store between CPU
processes), plus a helper that prepares single-host TPU launch variables. Collectives and point-to-point
transfers are traced inside ``lumen.compile``, mirroring ``torch.distributed``'s names; each runs
asynchronously and ends in a ``device_wait``, which the plan places just before the first op that reads the
result. Every process issues its collectives in program order, so a pair's sends and receives match in the
order they were written. ``group`` is ``None`` (every process), a ``DeviceMesh`` or a list of global ranks;
``src``, ``dst`` and roots are global ranks."""

import atexit
import ctypes
import importlib.util
import os
import socket
import subprocess
import sys

from lumen.distributed._C import (
    DeviceMesh,
    TCPStore,
    _barrier,
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
    "all_to_all",
    "barrier",
    "broadcast",
    "destroy_process_group",
    "gather",
    "get_rank",
    "get_world_size",
    "init_device_mesh",
    "init_process_group",
    "is_initialized",
    "recv",
    "reduce",
    "reduce_scatter",
    "scatter",
    "send",
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


@atexit.register
def _leave():
    """Holds an exiting process of the default group until every process exits: rank 0's store serves the
    others' collectives and rendezvous, so it must outlive their last request."""
    if is_initialized():
        _barrier(list(range(get_world_size())))


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


def _split(tensor, dim, n, op):
    """``tensor`` with ``dim`` split into ``n`` equal blocks along a new leading axis, block ``r`` its row ``r``;
    ``op`` names the collective in the error when ``dim`` doesn't split."""
    from lumen.graph import prims

    if tensor.shape[dim] % n:
        raise ValueError(f"{op}(): dimension {dim} of size {tensor.shape[dim]} doesn't split over {n} processes")
    split = prims.reshape(tensor, [*tensor.shape[:dim], n, tensor.shape[dim] // n, *tensor.shape[dim + 1 :]])
    order = [dim, *range(dim), *range(dim + 1, tensor.ndim + 1)]
    return prims.transpose(split, order) if dim else split


def _merge(rows, dim):
    """``rows``' leading axis (one row per process) concatenated along ``dim`` of a row: row ``r`` comes
    ``r``-th."""
    from lumen.graph import prims

    n, shape = rows.shape[0], rows.shape[1:]
    # The leading axis goes just before `dim`, then merges with it.
    order = [*range(1, dim + 1), 0, *range(dim + 1, len(shape) + 1)]
    moved = prims.transpose(rows, order) if dim else rows
    return prims.reshape(moved, [*shape[:dim], n * shape[dim], *shape[dim + 1 :]])


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
    return _merge(prims.device_wait(prims.all_gather(tensor, ranks)), dim % tensor.ndim)


def reduce_scatter(tensor, dim=0, op="sum", group=None):
    """``tensor`` combined across ``group``'s processes with ``op``, each keeping its share of ``dim`` (split
    into one equal block per process, in rank order): PyTorch's ``reduce_scatter_tensor``, any ``dim``."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    rows = _split(tensor, dim % tensor.ndim, len(ranks), "reduce_scatter")
    return prims.device_wait(prims.reduce_scatter(rows, op, ranks))


def all_to_all(tensor, split_dim=0, concat_dim=0, group=None):
    """``tensor`` split along ``split_dim`` into one equal block per process of ``group``, block ``j`` sent to
    the ``j``-th; the blocks received concatenated along ``concat_dim`` in rank order (PyTorch's
    ``all_to_all_single`` with equal splits, any dims)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    rows = _split(tensor, split_dim % tensor.ndim, len(ranks), "all_to_all")
    return _merge(prims.device_wait(prims.all_to_all(rows, ranks)), concat_dim % tensor.ndim)


def broadcast(tensor, src=0, group=None):
    """Process ``src``'s ``tensor``, given to every process of ``group`` (PyTorch's ``broadcast``)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    return tensor if len(ranks) == 1 else prims.device_wait(prims.broadcast(tensor, src, ranks))


def reduce(tensor, dst=0, op="sum", group=None):
    """``tensor`` combined across ``group``'s processes with ``op`` on process ``dst``; zeros on the others
    (PyTorch's ``reduce``)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    return tensor if len(ranks) == 1 else prims.device_wait(prims.reduce(tensor, op, dst, ranks))


def gather(tensor, dst=0, dim=0, group=None):
    """Every process in ``group``'s ``tensor`` concatenated along ``dim``, in rank order, on process ``dst``;
    zeros on the others (PyTorch's ``gather`` into one tensor)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    return _merge(prims.device_wait(prims.gather_to(tensor, dst, ranks)), dim % tensor.ndim)


def scatter(tensor, src=0, dim=0, group=None):
    """Process ``src``'s ``tensor`` split along ``dim`` into one equal block per process of ``group``, the
    ``r``-th going to the ``r``-th; the other processes pass a tensor of the same type, which is not read
    (PyTorch's ``scatter`` from one tensor)."""
    from lumen.graph import prims

    ranks = _ranks(group)
    if len(ranks) == 1:
        return tensor
    rows = _split(tensor, dim % tensor.ndim, len(ranks), "scatter")
    return prims.device_wait(prims.scatter_from(rows, src, ranks))


def send(tensor, dst, group=None):
    """Sends ``tensor`` to process ``dst``, which receives it with ``recv`` (PyTorch's ``send``). The send runs
    though nothing reads it; a process's sends and receives pair with another's in program order, and a send
    may wait for its receive, so two processes exchanging tensors order them oppositely (one sends first)."""
    from lumen.graph import prims, tracer

    tracer.keep(prims.device_wait(prims.send(tensor, dst, _ranks(group))))


def recv(tensor, src, group=None):
    """The tensor process ``src`` sends (``send``), of ``tensor``'s dtype and shape, whose values are not read
    (PyTorch's ``recv`` into ``tensor``, returned instead of written)."""
    from lumen.graph import prims

    return prims.device_wait(prims.recv(tensor.shape, tensor.dtype, src, _ranks(group)))


def barrier(group=None):
    """Blocks the host until every process of ``group`` has called ``barrier`` as many times, through the default
    group's store (PyTorch's ``barrier``); device work already queued is not waited for. Called outside traces."""
    _barrier(_ranks(group))
