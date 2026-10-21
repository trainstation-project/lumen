# `lumen/distributed`

`lumen.distributed`, after `torch.distributed`: one process per device
(launched by `torchrun` or the like), each knowing its `RANK` among
`WORLD_SIZE` processes. This folder holds the `DeviceMesh`, the n-d layout
of those ranks that parallelism is written against (`mesh["dp"]`,
`mesh.get_local_rank("tp")`). Process groups and collectives come later; a
mesh today is a label, like `lumen.device`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `get_rank` / `get_world_size` (the launcher's `RANK` / `WORLD_SIZE`, defaulting to one process) and `DeviceMesh`: a device type, a row-major array of global ranks, optional dimension names and this process's rank; `coordinate`, `local_rank`, `slice`. |
| `python.rs` | Bindings into `lumen.distributed._C`: the `DeviceMesh` class, `init_device_mesh`, `get_rank`, `get_world_size`. |
| `__init__.py` | The `lumen.distributed` package. |
| `_C.pyi` | Type stub for `python.rs`. |
| `tests.rs` | Rust tests for coordinates, slicing and validation. Python tests are `tests/test_distributed.py`. |

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
