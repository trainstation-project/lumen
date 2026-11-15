# tensor

The concrete tensor: a strided view of device storage, its dtypes and scalars, parameters (weights placed per device), and the Python, NumPy and DLPack interop. Tensors are what a compiled plan reads and writes; inside a trace, Python code works on `TracedTensor` (`lumen/graph/tracer.py`) instead.

## Files

- `mod.rs` declares the submodules and re-exports `Tensor` and helpers such as `contiguous_strides`.
- `tensor.rs` is `Tensor`: an `Arc<Storage>` plus dtype, shape, element strides and element offset. Cloning shares storage. It has factories (`zeros`, `ones`, `full`, `arange`, `from_slice`), views (`reshape`, `narrow`, `select`, `transpose`, `permute`, `squeeze_dim`, `unsqueeze`), copies across devices (`to`, `to_dtype`, `contiguous`) and host reads (`to_vec`). Byte size and alignment are per device: TPU tensors use the tiled layout's size and alignment.
- `storage.rs` is `Storage`: one device allocation (a `DataPtr` from `lumen/allocator`) with a unique id, plus its parameter state.
- `dtype.rs` holds `DType` (bool, unsigned and signed integers up to 64 bits, f16, bf16, f32, f64, 2- and 4-bit integers stored one per byte, and three float8 formats), the `Element` trait mapping Rust types to dtypes, the sub-byte and float8 storage types, and the `dispatch_dtype!` macro for running generic code on a runtime dtype.
- `scalar.rs` is `Scalar`, a dtype-less bool, integer or float used for fills and constants.
- `tensor_options.rs` is `TensorOptions`, an optional dtype and device; `DEFAULT_DTYPE` is f32.
- `parameter.rs` handles parameters: whole contiguous meta tensors that get a real tensor per device the first time a compiled function uses them (`placed`), optionally packed with other parameters into one shared block (`pack`). A version counter bumps on each write so compiled functions can tell their copies are stale.
- `dlpack.rs` and `dlpack.py` export tensors as DLPack capsules (versioned or legacy) that share memory, and adopt capsules from other libraries (`from_dlpack`), releasing the producer when the storage is freed.
- `python.rs` binds `Tensor` to Python (construction from nested lists, views, copies, element access, DLPack) and has `parse_dtype` and `dtype_name`.
- `__init__.py` exposes dtype names, `default_dtype`, factories (`tensor`, `empty`, `zeros`, `ones`, `full`, `arange`) and NumPy conversion (`to_numpy`, `from_numpy`, with `ml_dtypes` for the sub-byte and float8 dtypes).
- `_C.pyi` types the native module; `tests.rs` holds unit tests.

## How it connects

- Inside a trace, `zeros`, `ones`, `full` and `arange` bind `full`, `broadcast_in_dim` or `iota` primitives instead of allocating.
- `lumen.compile` places `lumen.nn.Module` weights through `parameter.rs`; merged dots read packed blocks built by `Parameter::pack`.
- `lumen/safetensors` loads into and saves from these tensors.

## Conventions

- `empty` returns uninitialized memory; reading it before writing is undefined behaviour in the Rust core.
- Only a whole contiguous meta tensor can be a parameter.
- Shapes and strides are in elements, not bytes; sub-byte dtypes still take a whole byte per element in memory.
