# `lumen/stream`

Streams: the per-device queue that ops submit work to without waiting, and
the means to wait for it. In Rust this is a one-method `Stream` trait plus a
backend per device; in Python it is `lumen.Stream`, modeled on
`torch.Stream`. Only MPS has real queueing logic in lumen itself (see
[`mps/`](mps/README.md)); CUDA wraps `cudart` synchronization, and CPU/meta
streams are no-ops because their work is already done when the op returns.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `Stream` trait (`synchronize`) and `stream_for(device)`, which calls `device.check()` and returns the device's stream. |
| `cpu.rs` | `CpuStream` — `synchronize` is a no-op. |
| `meta.rs` | `MetaStream` — `synchronize` is a no-op. |
| `cuda.rs` | `CudaStream { index }` and `cudart` FFI: `synchronize` (`cudaDeviceSynchronize`), `synchronize_external` (`cudaStreamSynchronize` on a foreign stream), `wait` (event record on the default stream + `cudaStreamWaitEvent`). Only with `cfg(lumen_cuda_linked)`. |
| `python.rs` | The `lumen.Stream` pyclass (`PyStream`), registered into `lumen._C`. |
| `__init__.py` | Re-exports `Stream` from `lumen._C`; `lumen/__init__.py` exposes it as `lumen.Stream`. |
| `_C.pyi` | Type stub for `Stream`. |
| `tests.rs` | Every available device has a synchronizable stream; unavailable devices get an error. |

## Subfolders

[`mps/`](mps/README.md) — the Metal command-queue stream (Rust + Objective-C++).

## Rust API

```rust
pub trait Stream: Send + Sync { fn synchronize(&self); }
pub fn stream_for(device: Device) -> Result<Arc<dyn Stream>, String>;
```

`stream_for` returns the device module's error string for an unavailable
device. Variants for backends not compiled in are unreachable, since such a
device never passes `check()`.

## `lumen.Stream` (Python)

* `Stream(device=None, *, stream_id=0)` — `device` goes through
  `resolve_device`. A nonzero `stream_id` (a foreign stream handle) is only
  accepted for CUDA; other devices raise `ValueError`.
* `synchronize()` — recorded as a `synchronize` op in the profiler; releases
  the GIL. For CUDA with `stream_id != 0` it synchronizes that external
  stream, otherwise the device's lumen stream.
* `wait_stream(other)` — no-op for the same stream. If `other` is lumen's
  CUDA stream on the same device, makes `self`'s external stream wait on it
  through a CUDA event (no host block); otherwise blocks the host until
  `other` has finished.
* `command_buffer()` / `buffer(tensor)` — MPS only: the raw pointer of the
  current `MTLCommandBuffer`, and a tensor's `(MTLBuffer pointer, byte
  offset)`, for interop with external Metal code. Raise on other devices.
* `__eq__` / `__hash__` on `(device, stream_id)`; frozen pyclass.

## Gotchas

* There is one lumen stream per device; `stream_id` only names external CUDA
  streams, it does not create streams.
* CUDA `synchronize` is device-wide (`cudaDeviceSynchronize`), not
  per-stream.
* `buffer()` returns pointers that are unretained; the allocator keeps the
  `MTLBuffer` alive, so do not use them after the tensor is freed.

## Related

* [`../device`](../device/README.md) — `Device`, `check()`, `resolve_device`.
* [`../allocator`](../allocator/README.md) — the MPS allocator synchronizes
  the stream before declaring out-of-memory.
* [`../profiler`](../profiler/README.md) — `stop()` synchronizes MPS.
* `tests/test_stream.py` — Python-side tests.
