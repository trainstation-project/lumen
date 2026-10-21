# `lumen/cute_dsl_utils`

Helpers for writing CUDA kernels in NVIDIA's CuTe DSL (`cutlass.cute`, from
the `nvidia-cutlass-dsl` dependency, Linux only): dtype mapping, tensor
conversion, fast activations, and generic grid-stride elementwise kernel
classes. Pure Python; importing it requires `cutlass` and `cuda.bindings`.
No other module in the repo imports this package at present.

## Files

| File | Contents |
| --- | --- |
| `__init__.py` | Re-exports the public helpers listed below. |
| `constants.py` | `get_cute_dtype(name)`: lumen dtype name (`float32`, `float16`, `bfloat16`, `int32`, `int64`, `uint32`) → CuTe dtype. |
| `utils.py` | `get_powers_of_2`, `get_fake_cute_tensor` (compile-time stand-in with symbolic strides), `element_bytes`, `get_alignment` (largest of 4/8/16 dividing the address), `tensor_to_cute_tensor` (DLPack → CuTe tensor with dynamic layout). |
| `activations.py` | `@cute.jit` `tanh` (fastmath, computed in float32) and `sigmoid` (`0.5 * tanh(0.5x) + 0.5`). |
| `boundary.py` | `lane_boundary`: per-thread copy slice and predicate mask for a tile that may run past the tensor's shape. |
| `elementwise.py` | `ElementwiseCUDAKernel` (subclass and implement `compute(xs) -> ys`) and `get_compiled_elementwise_cuda_kernel` (compile-once cache keyed on the caller). |
| `packed_elementwise.py` | `ElementwisePackedCUDAKernel`: like the above but over two tensor groups, the first tiled at half the vector width. |

## Elementwise kernels

* `__call__(mXs, mYs, stream)` tiles all tensors with the same thread/value
  layout (`BLOCK_SIZE` threads as warps of 32, `M` rows of 128-bit vectors),
  launches one block per SM, and each block strides through tiles
  (`BLOCK_ID += NUM_BLOCKS`), so one compiled kernel serves any size.
* Full tiles copy unpredicated; the tail tile uses the `lane_boundary` mask.
* `get_compiled_elementwise_cuda_kernel` builds fake tensors with shape
  `(sym, sym(divisibility))` and compiles with `--enable-tvm-ffi`, caching
  in `caller_op.cache[key]`.

## Gotchas

* `packed_elementwise.py` uses `WARP_SIZE` and `LOG_WARP_SIZE` without
  importing them from `elementwise.py`, so `ElementwisePackedCUDAKernel.__call__`
  would hit a `NameError` as written.
* `tensor_to_cute_tensor` treats a stride-0 (broadcast) leading dimension as
  no leading dimension.

## Related

* `pyproject.toml` — the `nvidia-cutlass-dsl` dependency.
