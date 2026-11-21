# tensor

The concrete tensor: a strided view of device storage, its dtypes and scalars, parameters (weights placed per device), and the Python, NumPy and DLPack interop. Tensors are what a compiled plan reads and writes; inside a trace, Python code works on `TracedTensor` (`lumen/graph/tracer.py`) instead.

## Files

- `mod.rs` declares the submodules and re-exports `Tensor` and helpers such as `contiguous_strides`.
- `tensor.rs` is `Tensor`: an `Arc<Storage>` plus dtype, shape, element strides and element offset. Cloning shares storage. It has factories (`zeros`, `ones`, `full`, `arange`, `from_slice`), views (`reshape`, `narrow`, `select`, `transpose`, `permute`, `squeeze_dim`, `unsqueeze`), copies across devices (`to`, `to_dtype`, `contiguous`) and host reads (`to_vec`). Byte size and alignment are per device: TPU tensors use the tiled layout's size and alignment. A TPU tensor's view that changes its tiled layout (a transpose, a narrow, a reshape across tiles) is copied through the host into a contiguous TPU tensor when taken (`tiled`), since no TPU kernel reads a strided view.
- `storage.rs` is `Storage`: one device allocation (a `DataPtr` from `lumen/allocator`) with a unique id, plus its parameter state.
- `dtype.rs` holds `DType` (bool, unsigned and signed integers up to 64 bits, f16, bf16, f32, f64, 2- and 4-bit integers stored one per byte, and three float8 formats), the `Element` trait mapping Rust types to dtypes, the sub-byte and float8 storage types, and the `dispatch_dtype!` macro for running generic code on a runtime dtype.
- `scalar.rs` is `Scalar`, a dtype-less bool, integer or float used for fills and constants.
- `tensor_options.rs` is `TensorOptions`, an optional dtype and device; `DEFAULT_DTYPE` is f32.
- `parameter.rs` handles parameters: whole contiguous meta tensors that get a real tensor per device the first time a compiled function uses them (`placed`), optionally packed with other parameters into one shared block (`pack`). A version counter bumps on each write so compiled functions can tell their copies are stale.
- `dlpack.rs` and `dlpack.py` export tensors as DLPack capsules (versioned or legacy) that share memory, and adopt capsules from other libraries (`from_dlpack`), releasing the producer when the storage is freed.
- `python.rs` binds `Tensor` to Python (construction from nested lists, views, copies, element access, DLPack) and has `parse_dtype` and `dtype_name`. Its `PyTensor` also holds the optional axis names (`names`, `_with_names`; part of `_signature`), which copies (`to`, `cpu`, `clone`, `contiguous`) keep and views drop, and the optional placement of a sharded tensor (`placement`, `_with_placement`: a `Sharding` and the global shape; part of `_signature`), which copies keep, whose `shape` and `numel` are the global ones, and whose views raise (they would index the local shard).
- `names.py` holds `AxisType` (`str | int`: a position or an axis name) and `Axes` (one or a sequence of them), the types of every dimension argument, and `axis`, the one resolver of an `AxisType` to a position (an `IndexError` out of range, a `KeyError` for an unknown name) behind every op and collective taking a dimension; it also checks names and resolves orders (`check`, `position`, `renamed`, `refined`, `order`), for `Tensor` and `TracedTensor` alike.
- `placements.py` holds the placements of a sharded tensor, PyTorch DTensor's: `Shard(dim)` (an `AxisType`, so `Shard("seq")`), `Replicate()`, `Partial(reduce_op)`, and `Sharding` (a mesh and one placement per mesh dim, `Shard` dims resolved to positions by `sharding`), with the shape math `Tensor` and `TracedTensor` share (`chunk`, `local_shape`, `block_shape`, `global_shape`, `even`, this process's `coordinate`, `group` and `offsets`). A sharded dim splits as `torch.chunk` does (DTensor's): blocks of `ceil(size / n)`, the last ones shorter or empty, so local shapes may differ between processes; `block_shape` is the largest, the same on every process, for choices that must agree. A dim may be sharded over several mesh dims, each splitting the block the ones before it left.
- `__init__.py` exposes dtype names, `default_dtype`, factories (`tensor`, `empty`, `zeros`, `ones`, `full`, `arange`) and NumPy conversion (`to_numpy`, `from_numpy`, with `ml_dtypes` for the sub-byte and float8 dtypes), and attaches the methods of names to `Tensor` (`rename`, `refine_names`, `align_to`, `size`, `dim_of`) and of placements (`distribute`, `to_local`, `placements`, `device_mesh`; `full_tensor` and `redistribute` record collectives, so they run on traced tensors only).
- `_C.pyi` types the native module; `tests.rs` holds unit tests.

## How it connects

- Inside a trace, `zeros`, `ones`, `full` and `arange` bind `full`, `broadcast_in_dim` or `iota` primitives instead of allocating.
- `lumen.compile` places `lumen.nn.Module` weights through `parameter.rs`; merged dots read packed blocks built by `Parameter::pack`.
- `lumen/safetensors` loads into and saves from these tensors.

## Conventions

- `empty` returns uninitialized memory; reading it before writing is undefined behaviour in the Rust core.
- Only a whole contiguous meta tensor can be a parameter.
- Shapes and strides are in elements, not bytes; sub-byte dtypes still take a whole byte per element in memory.
