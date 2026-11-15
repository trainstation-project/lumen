# custom_op

User-defined ops. There are two mechanisms here: Python functions that mutate tensor arguments and are called from compiled code as a `custom_call` node, and Python kernel factories registered for built-in ops that accept them, looked up by op name and device at dispatch time. It also lets users compile their own Metal kernels and launch them on lumen's MPS stream.

## Files

- `__init__.py` is the Python API: `custom_op` (decorator) and `CustomOp`, `register`, `unregister`, `registered_ops`, `signature`, and `MetalKernel`.
- `mod.rs` declares the submodules and `vector_size`, the widest vector width (up to `MAX_VECTOR_BYTES`) a launcher can use on a tensor given its strides and address.
- `register.rs` keeps the registered Python kernel factories. On a launch it compiles a launcher per (handle, dtype, shape, strides, vector size), caches it, and calls it with the device's default stream. A launcher that exposes a TVM FFI function is called natively without the GIL; any other callable is called under the GIL.
- `ffi.rs` opens `libtvm_ffi` with `dlopen` and calls TVM FFI function handles directly, turning a raised error into `"kind: message"`.
- `mps.rs` holds `MetalKernel` (Metal source compiled at runtime into a pipeline cached in the MPS kernel cache) and `Grid`, the launch size. Without MPS, compiling fails and launching is a no-op.
- `python.rs` binds `MetalKernel` to Python.
- `dummy_op.rs` is `dummy_op`, a fill-like op with no built-in kernels, used to test Python kernel registration.
- `tests.rs` holds the unit tests.
- `_C.pyi` types the native module.

## How it connects

- `CustomOp.__call__` binds `prims._custom_call` into the current trace (`lumen/graph`). It registers a `run` closure with `lumen.graph._C._register_custom_op`; at run time `lumen/graph/custom.rs` calls back into it by kernel number.
- The node's results are copied back into the mutated arguments with `copy_`, so the program sees the mutation.
- Kernel tables are `Op` entries keyed by `DispatchKey` from `lumen/runtime/dispatch`.

## Conventions

- A custom op returns nothing; its results are exactly the arguments named in `mutates_args`, which must be tensors and must be complete (an unnamed mutation is not respected).
- `overlappable` maps a mutated argument to read arguments whose memory it may share; the names are checked against the signature.
- Each tensor must be its own argument, not inside a list, tuple or dict.
- Calling a custom op outside `lumen.compile` raises.
- Python kernels can only be registered on non-CPU devices and only for ops listed by `registered_ops()`.
