# `lumen/tensor`

The concrete tensor: a strided view over a device buffer (PyTorch's
`TensorImpl` over `StorageImpl`), plus dtypes, scalars, factory options,
parameter placement, DLPack interop and the Python bindings.

A `Tensor` here *holds data*; it does not compute. lumen always runs inside
`lumen.compile`: a compiled function sees `TracedTensor`s
(`lumen/graph/tracer.py`), records primitives into a `Graph`, and the
compiled `Plan` reads and writes these tensors. What a `Tensor` can do on
its own is metadata, views, element access, copies between devices, and
fills (on the host, then copied in; on CUDA a generated Triton kernel); even a dtype conversion (`to_dtype`) builds a one-node `Cast` graph
and runs it through `graph_compiler::compile`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module wiring; re-exports `Tensor` (path `lumen::tensor::Tensor`) and crate-internal stride helpers. |
| `tensor.rs` | `Tensor { storage: Arc<Storage>, dtype, shape, strides, offset }`: factories (`zeros`, `ones`, `full`, `arange`, `from_slice`, unsafe `empty`), views (`reshape`, `narrow`, `select`, `transpose`, `permute`, `squeeze_dim`, `unsqueeze`), `contiguous`, `to(device)`, `to_dtype`, `to_vec`, `get`/`set`, `fill_`, `copy_`, `zero_`, `wait_`, `Display`; `contiguous_strides`, `flat_offset`, `for_each_index`. `fill_`, `zeros` and `ones` fill a host tensor and `copy_` it in, except on CUDA, where a generated Triton kernel fills it on the GPU (`graph_compiler/triton`'s `fill`); on CUDA `copy_` from the same GPU and `contiguous` are a generated Triton copy, any strides on either side (`graph_compiler/triton`'s `copy`). |
| `storage.rs` | `Storage`: an owning byte buffer on a device (`DataPtr` from the device's allocator), a unique `id`, and the meta-tensor `Parameter` state. |
| `dtype.rs` | `DType` (Bool, U8–U64, I8–I64, F16, BF16, F32, F64, and JAX's narrow I2/I4/U2/U4, F8E4M3FN, F8E5M2, F8E4M3B11FNUZ), the `Element` trait for the Rust element types (`half::f16`/`bf16` re-exported; the narrow ones a byte each, as `ml_dtypes` keeps them: `i4`, `f8e4m3fn`, ...), and the `dispatch_dtype!` macro. |
| `scalar.rs` | `Scalar { Bool, Int, Float }` (`c10::Scalar`): `full`'s value and `arange`'s end; `inferred_dtype` (bool / int64 / float32). |
| `tensor_options.rs` | `TensorOptions` (dtype + device builder) and `DEFAULT_DTYPE = F32`. |
| `parameter.rs` | Weights as meta tensors: `is_parameter`, `placed(device)`, `pack(members, dimension, interleave, device)` (side by side, or interleaved: element j of the i-th at `j * interleave + i`), `version` / `mark_written`, `copy_into_placements`. |
| `dlpack.rs` | DLPack export (legacy and versioned capsules) and import (`python` feature). |
| `python.rs` | The `lumen.tensor._C.Tensor` pyclass (`PyTensor`), dtype-string parsing, nested-list construction, `_pack`, `_to_dlpack*`, `_from_dlpack`. |
| `__init__.py` | Python package: dtype constants (strings like `"float32"`), `default_dtype`, factories (`tensor`, `empty`, `zeros`, `ones`, `full`, `arange`), `to_numpy` / `from_numpy`; attaches `__array__` / `__dlpack__` to `Tensor`. All re-exported at `lumen.*`. |
| `dlpack.py` | `__dlpack__` / `__dlpack_device__` (CPU, CUDA, Metal), CUDA stream handshake, and `from_dlpack`. |
| `_C.pyi` | Type stub for what `python.rs` defines. |
| `tests.rs` | Rust unit tests (dtypes, factories, views, copies, parameters, ...). |

## Key concepts

* **Views share storage.** `reshape` (when contiguous), `narrow`, `select`,
  `transpose`, `permute`, `(un)squeeze` only change shape/strides/offset.
  `storage_id` / `shares_storage_with` expose aliasing.
* **Device copies** go through `kernels::device_transfer` (`copy_h2d` /
  `copy_d2h`); only CPU storage is dereferenced directly. Device-to-device
  copies hop through the CPU. Meta tensors have no data: reading or copying
  one panics / raises.
* **Parameters are meta tensors.** A whole, contiguous meta tensor
  (`is_parameter`) is a weight. When a compiled function taking it compiles,
  `tracer.py` calls `_placed(device)` to get (or zero-allocate) its memory on
  that device, shared by every compiled function. `_pack` places several
  weights as slices of one concatenated block (for the weights a plan
  lists in `plan.packed`, e.g. merged dots), or
  returns `None` if they are already placed differently. `copy_` into a
  parameter writes all its placements and bumps `version`, which the tracer
  uses to notice writes; it fails if the parameter was never placed.
* **Factories inside a trace** (`lumen.zeros`, `ones`, `full`, `arange` in
  `__init__.py`) record a `full` / `iota` primitive instead of allocating, and
  ignore `device`.
* `empty` is uninitialized: reading before writing is UB in the Rust core.

## Invariants & gotchas

* Dtypes are named by strings on the Python side (`"bfloat16"`, `"int64"`);
  `parse_dtype` also accepts short forms (`"f32"`, `"bf16"`, `"half"`).
* `to_numpy` refuses bfloat16 (NumPy has none); it copies to the host once
  and reads through DLPack.
* `lumen.tensor` is the factory function, not this package (as in torch).
* Within `__init__.py`, `bool` shadows the builtin (like `torch.bool`).

## Related

* [`lumen/allocator`](../allocator/README.md) — where `Storage` gets its bytes.
* [`lumen/device`](../device/README.md) — `Device` and `resolve_device`.
* [`lumen/kernels`](../kernels/) — copies and waits on tensors.
* [`lumen/graph`](../graph/) — `TracedTensor`, the compute-side tensor.
* `tests/test_tensor.py`, `tests/test_dlpack.py`, `tests/device/test_tensor_devices.py`.
