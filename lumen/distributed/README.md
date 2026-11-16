# distributed

Multi-process support: the default process group, TCP rendezvous store, device meshes, barriers, the collectives and point-to-point transfers a plan runs, and JAX-style `shard_map` over a mesh. Collectives are traced inside `lumen.compile` as primitives (`prims.all_reduce`, `prims.send`, ...) and executed by the plan's steps on the CPU or CUDA.

## Files

- `__init__.py` is the Python API, mirroring `torch.distributed` names: `init_process_group`, `init_device_mesh`, `DeviceMesh`, `TCPStore`, `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `reduce`, `gather`, `scatter`, `send`, `recv`, `barrier`, and `set_tpu_launch_env`. Each collective binds its primitive followed by `prims.device_wait`. It also loads NCCL's library from the `nvidia-nccl` wheel when present.
- `spmd.py` is JAX's `shard_map` for lumen's one-process-per-device model: `shard_map`, `PartitionSpec` (`P`) and the named collectives a mapped function calls, with `jax.lax`'s names and arguments: `psum`, `pmean`, `pmax`, `pmin`, `all_gather`, `psum_scatter`, `ppermute`, `all_to_all`, `axis_index` and `axis_size`. `lumen.distributed` exports `shard_map`, `P` and `PartitionSpec`; the named collectives stay in `lumen.distributed.spmd` (as JAX's live in `jax.lax`), since `all_gather` and `all_to_all` already name the torch-style collectives here.
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
- `spmd.py` lowers named collectives to the functions in `__init__.py` over `mesh[axes]`, `ppermute` to `send`/`recv`, and its boundaries (`_Shard`, `_Unshard`) and `ppermute` to `lumen.autograd.Function`s with hand-written backwards.

## Conventions

- `group` is `None` (every process), a `DeviceMesh` or a list of global ranks; `src`, `dst` and roots are global ranks.
- Every process issues its collectives in program order, so a pair's sends and receives match in the order they were written.
- Collectives are asynchronous and end in a `device_wait`, which the plan places just before the first op that reads the result. A collective over one rank returns its input unchanged.
- Messages between a pair over the store are keyed by the pair and a running count, so they arrive in send order.
- CSE (`lumen/graph_compiler/passes/cse.rs`) never merges collectives: two that are equal on one process (shard_map blocks at the same offset) may differ on another, which would then send one more message.

## shard_map

- `shard_map(f, mesh=, in_specs=, out_specs=)` (or `@shard_map(mesh=..., ...)`) follows JAX's global view. The mapped function takes the global tensors, which every process holds; each process slices out its block (a `slice` in the trace) per `in_specs` from its mesh coordinate, runs `f` on it, and assembles the outputs per `out_specs`: a dimension naming axes is `all_gather`ed over their sub-group, and an axis no dimension names is taken as replicated (not checked) and this process's block returned. Every process returns the global outputs.
- A spec entry is an axis name, a tuple of names (the first outermost, as in `axis_index`) or None; specs are pytree prefixes (one spec for all arguments, or one per argument). The mesh needs `mesh_dim_names`, and each split dimension must divide evenly.
- Axis names resolve to the innermost `shard_map`'s mesh; using one outside a mapped function, or one its mesh lacks, raises `NameError` naming it. `axis_index` and `axis_size` are Python ints: each process traces its own program.
- `ppermute` runs over `send`/`recv`, ordered so blocking sends never deadlock (around a cycle its smallest index sends first); a process no pair sends to gets zeros. Its receive is kept even when nothing reads it, so messages stay paired under `grad`.
- Gradients follow JAX's `shard_map` transpose without replication checks: an output's cotangent (the same on every process) is sliced to each block and divided by the number of copies along its unnamed axes; an input block's cotangent is summed along the input's unnamed axes and gathered back, so every process gets the full gradient. Inside `f`, `psum`, `pmean`, `all_gather`, `psum_scatter`, `all_to_all` and `ppermute` differentiate; `pmax` and `pmin` don't. The loss after the shard_map must be computed the same on every process.
- Tested on CPU processes over the store; CUDA uses the same primitives through NCCL.
