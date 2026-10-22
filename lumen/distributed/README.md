# `lumen/distributed`

`lumen.distributed`, after `torch.distributed`: one process per device
(launched by `torchrun` or the like), each knowing its `RANK` among
`WORLD_SIZE` processes. This folder holds the `DeviceMesh`, the n-d layout
of those ranks that parallelism is written against (`mesh["dp"]`,
`mesh.get_local_rank("tp")`), and the rendezvous through which the processes
find each other (`init_process_group`). Collectives come later.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The default process group: `rendezvous` (`env://`, `tcp://host:port`), `init_process_group` (a barrier on the store), `destroy_process_group`, `is_initialized`; `get_rank` / `get_world_size` (the group's, else the launcher's `RANK` / `WORLD_SIZE`, defaulting to one process); and `DeviceMesh`: a device type, a row-major array of global ranks, optional dimension names and this process's rank; `coordinate`, `local_rank`, `slice`. |
| `backend.rs` | `MeshBackend`, a device's part in a mesh, implemented by each `DeviceBackend` (`Cpu`, `Mps`, `Cuda`, `Tpu`, `Meta`): `world(mesh_size)`, the rank and world a new mesh is built in. By default it rendezvous when the world has several processes; `Mps` overrides it, and `Tpu` first joins its process to the TPU slice (`tpu_slice_env`). `mesh_backend(device_type)` picks one. |
| `store.rs` | `TcpStore`, PyTorch's `TCPStore`: a key-value store served by one process (the master, a thread per connection) that every process connects to; `set`, `get` (waits for the key), `add` (a counter), `wait`. |
| `python.rs` | Bindings into `lumen.distributed._C`: `DeviceMesh`, `init_device_mesh`, `TCPStore`, `init_process_group`, `destroy_process_group`, `is_initialized`, `get_rank`, `get_world_size`. |
| `__init__.py` | The `lumen.distributed` package, and `set_tpu_launch_env`, the launcher's step for TPUs. |
| `__main__.py` | `python -m lumen.distributed`: `set_tpu_launch_env()`'s variables, as `NAME="value"` lines to export before `torchrun`. |
| `_C.pyi` | Type stub for `python.rs`. |
| `tests.rs` | Rust tests for coordinates, slicing, validation and the store. Python tests are `tests/test_distributed.py`. |

## Rendezvous

```python
lumen.distributed.init_process_group()  # env://: MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
lumen.distributed.init_process_group(init_method="tcp://10.0.0.1:29500", rank=r, world_size=n)
lumen.distributed.init_process_group(store=store, rank=r, world_size=n)
```

As in PyTorch, rank 0 serves a `TCPStore` at the master's address and every
rank connects to it (retrying until the timeout, 30 minutes by default, so
processes may start in any order). `init_process_group` counts the ranks in
on the store and returns once all `world_size` have arrived.

A mesh rendezvous by itself, as PyTorch's does: building one in a process
of a world of several (`WORLD_SIZE` > 1) with no process group yet runs
`init_process_group()` (`env://`). In a world of one there is no one to
meet. How a device takes part is its `MeshBackend`'s: an `"mps"` mesh never
rendezvous, since a Mac has one MPS device: the device must be there,
and the mesh must hold exactly it, in a world of one process. The store is
what later rendezvous build on: PJRT's distributed runtime is to read and write
its keys through it. Collective backends (`backend=`) come with the
collectives.

The wire format is lumen's own (length-prefixed keys and values), not
c10d's: a lumen store talks to lumen processes.

## TPUs: a process per chip, one slice

As in [torch_tpu](https://github.com/google-pytorch/torch_tpu)
(`docs/distributed_support.md`, `csrc/common/environment.cc`). A TPU chip
is a VFIO device a single process can open, and libtpu opens every chip it
sees, so each process sees only its own (`TPU_VISIBLE_CHIPS`): it is that
process's `tpu:0`. But the processes form one slice, not a TPU each: libtpu
is told the slice's shape (`TPU_PROCESS_BOUNDS`, e.g. `2,2,1` for a v6e-4),
one chip per process, and every process's address
(`TPU_PROCESS_ADDRESSES`), and its slice builder connects them when the
client is created (each waits for all). Programs then run on the slice's
devices, which collectives over ICI need.

The processes must agree on the slice, so the launcher decides it once:

```bash
eval $(python -m lumen.distributed | sed 's/^/export /')
torchrun --nproc-per-node $WORLD_SIZE train.py
```

`set_tpu_launch_env(nproc_per_node=None)` (what that prints) sets
`LUMEN_TPU_TOPOLOGY` (a single host's: 1, 4 or 8 chips are `1,1,1`,
`2,2,1`, `2,4,1`; others must be given) and
`LUMEN_TPU_SLICEBUILDER_ADDRESSES` (`localhost:<free port>` per process),
counting the chips in a child process so the launcher holds none, and
keeps what is already set. In each worker, the first `"tpu"` mesh of a
world of several turns them, with `RANK` and `LOCAL_RANK`, into libtpu's
variables before the rendezvous. So the mesh comes before anything else
touches the TPU; a process that opened it first gets an error, since its
libtpu already runs on its own. A world larger than the addresses (several
slices, over DCN) is not supported yet.

## The mesh

```python
mesh = lumen.distributed.init_device_mesh("tpu", (2, 2), mesh_dim_names=("dp", "tp"))
mesh.mesh                  # [[0, 1], [2, 3]]: ranks, row-major
mesh.get_coordinate()      # this rank's index along each dimension
mesh.get_local_rank("tp")  # its index along one
mesh["tp"]                 # the 1-d submesh of "tp" holding this rank
```

As in PyTorch, every process builds the same mesh. `DeviceMesh(device_type,
mesh, mesh_dim_names=...)` takes any rectangular nested list of distinct
ranks below `WORLD_SIZE`; `init_device_mesh` lays `0..n` out row-major.
Indexing by names (`mesh["tp", "dp"]`, in any order) gives the submesh
through this rank, the other dimensions fixed at its coordinate; it needs
`mesh_dim_names`, and a rank outside the mesh has no submesh (nor
coordinate). PyTorch's `get_group`, `_flatten` and `from_group` wait for
process groups.

## Inside `lumen.compile`

A mesh is immutable and hashable, so a compiled function can take it as an
argument (a static one: a different mesh compiles another plan) or close
over it. Its queries are Python ints, folded into the graph as constants,
as `torch.compile` treats them: each process traces its own program, with
its own rank.
