# distributed

Multi-process support: the default process group, TCP rendezvous store, device meshes, barriers, and the collectives and point-to-point transfers a plan runs. Collectives are traced inside `lumen.compile` as primitives (`prims.all_reduce`, `prims.send`, ...) and executed by the plan's steps on the CPU or CUDA.

## Files

- `__init__.py` is the Python API, mirroring `torch.distributed` names: `init_process_group`, `init_device_mesh`, `DeviceMesh`, `TCPStore`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `reduce`, `gather`, `scatter`, `send`, `recv`, `barrier`, and `set_tpu_launch_env`. It also exports DTensor's placements (`Shard`, `Replicate`, `Partial`, from `lumen/tensor/placements.py`) and `from_local`, which places a process's shard over a mesh (given the global `shape` for uneven shards); ops on sharded tensors record their collectives through `lumen/graph/propagation/sharding.py`. Each collective binds its primitive followed by `prims.device_wait`. It also loads NCCL's library from the `nvidia-nccl` wheel when present.
- `__main__.py` prints TPU launch variables (`python -m lumen.distributed`) for a shell to export before starting workers.
- `mod.rs` holds the default process group (rank, world size, shared store), `env://` and `tcp://` rendezvous, `barrier`, and `DeviceMesh` (shape, dim names, coordinates, `slice`, `local_rank`).
- `store.rs` is `TcpStore`: a threaded key-value server on the master rank and a blocking client (`set`, `get`, `add`, `delete_key`, `wait`), with a small binary wire format.
- `backend.rs` holds `MeshBackend`, how each device type finds this process's rank and world size, and `tpu_slice_env`, the libtpu environment that makes one worker drive one chip of a multi-process slice.
- `collective.rs` defines the `Communicator` trait and runs every collective over it once. A backend provides sends, receives and the reducing collectives; gathers, scatters, broadcasts and all-to-alls default to grouped sends and receives.
- `nccl.rs` is the CUDA communicator: `libnccl.so.2` loaded at run time, one communicator per group of ranks and device, created from an id rank 0 shares through the store. Collectives run on a per-device NCCL stream; `device_wait` makes lumen's stream wait for them.
- `host.rs` is the CPU communicator: each message is a store key set by the sender and taken by the receiver, so it is meant for tests and small CPU worlds. It runs synchronously inside the plan step.
- `python.rs` binds `TCPStore`, `DeviceMesh` and the process group functions.
- `tests.rs` holds unit tests for meshes, backends, the TPU env, the store and the default collectives.

## How it connects

- CPU plan steps (`lumen/plan/step/cpu.rs`) run collectives through `Host` and `collective::run`; CUDA steps go through `nccl::run`.
- Autograd rules for the collectives (`lumen/autograd/rules.py`) only differentiate sums.

## Conventions

- `group` is `None` (every process), a `DeviceMesh` or a list of global ranks; `src`, `dst` and roots are global ranks.
- Every process issues its collectives in program order, so a pair's sends and receives match in the order they were written.
- Collectives are asynchronous and end in a `device_wait`, which the plan places just before the first op that reads the result. A collective over one rank returns its input unchanged.
- Messages between a pair over the store are keyed by the pair and a running count, so they arrive in send order.
