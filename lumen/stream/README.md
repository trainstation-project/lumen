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
| `tpu.rs` | `TpuStream { index }` — `synchronize` waits for the programs run on the device so far (`allocator::tpu::wait`): a TPU's kernels run in order on its memory, asynchronously (each takes the previous one's result, `lumen/kernels/tpu`), and every read of it (`copy_to_host`) waits for the latest itself. Only with `cfg(lumen_tpu_linked)`. |
| `cuda.rs` | `CudaStream { index }` and `cudart` FFI: `synchronize` (`cudaDeviceSynchronize`), `synchronize_external` (`cudaStreamSynchronize` on a foreign stream), `wait` (event record on the default stream + `cudaStreamWaitEvent`). Only with `cfg(lumen_cuda_linked)`. |
| `python.rs` | The `lumen.Stream` and `lumen.Event` pyclasses (`PyStream`, `PyEvent`), registered into `lumen.stream._C`. |
| `__init__.py` | Re-exports `Event` and `Stream` from `lumen.stream._C`; `lumen/__init__.py` exposes them as `lumen.Event` and `lumen.Stream`. |
| `_C.pyi` | Type stub for `Stream`. |
| `tests.rs` | Every available device has a synchronizable stream; unavailable devices get an error. |

## Subfolders

[`mps/`](mps/README.md) — the Metal command-queue stream (Rust + Objective-C++).

`event/` — events, as streams are laid out: the `Event` trait and
`event_for(device, timing)` (`mod.rs`), and one implementation a device
(`cpu.rs`, `mps.rs`, `tpu.rs`, `cuda.rs`). A device implements `record` and
`marks` (the `Recorded` times its marker started and ended, set when the
device reaches it); the trait's `query`, `synchronize` and `elapsed_time`
work from those. CUDA overrides them with `cudaEvent_t` calls, MPS's
`synchronize` commits the open command buffer first.

## Rust API

```rust
pub trait Stream: Send + Sync { fn synchronize(&self); }
pub fn stream_for(device: Device) -> Result<Arc<dyn Stream>, String>;
pub trait Event: Send + Sync { /* record, query, synchronize, elapsed_time */ }
pub fn event_for(device: Device, timing: bool) -> Result<Box<dyn Event>, String>;
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

## `lumen.Event` (Python)

Modeled on `torch.cuda.Event`.

* `Event(device=None, *, enable_timing=False)`; the meta device has none.
* `record()` marks the event after the work submitted to the device's
  stream so far; recording again moves it. `query()` is whether the work
  before it is done (an event never recorded is); `synchronize()` waits for
  it (releasing the GIL).
* `elapsed_time(end)` is the time from this event to `end`, in
  milliseconds, waiting for both; both on one device, recorded, with
  `enable_timing`.

Each device times an event its own way:

* **MPS**: `record()` encodes a one-thread kernel (`event_marker`) in its
  own timed compute encoder, whose start and end the GPU's timestamp
  counters sample (the command buffer's `GPUStartTime`/`GPUEndTime` where
  the GPU samples none). `elapsed_time` is from the end of the first
  marker to the start of the second: GPU time, without the markers. (A
  `torch.mps.Event` times on the host, with a floor near 100 µs.)
* **CUDA**: a `cudaEvent_t` on the default stream (`cudaEventRecord`,
  `cudaEventElapsedTime`).
* **TPU**: `record()` runs the kernels batched so far, then registers a
  callback on the result of the last program (`PJRT_Event_OnReady`); the
  event's time is the host clock when that program finished. Its
  resolution is the callback's latency, microseconds: it times programs,
  not single short kernels.
* **CPU**: an op's work is done when it returns, so `record()` reads the
  host clock.

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
