# `lumen/stream/mps`

lumen's single Metal stream: one `MTLCommandQueue`, one open command buffer
and compute encoder that every MPS kernel launch encodes into, batched
commits, completion callbacks that keep tensors alive and report GPU
timings to the profiler, and `synchronize`. Compiled only when `build.rs`
builds the Objective-C++ shim (`cfg(lumen_mps_linked)`, macOS); `mps.mm` is
part of the `lumen_mps_shim` static library.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Rust side: `MpsStream`, `synchronize`, `submit` / `cancel` / `completed`, `mark`, `flush`, `encoded`, `command_buffer`. |
| `mps.h` | C/ObjC interface used by the kernel shims (`ops/mps/shim.mm`, `ops/fill/mps/shim.mm`): `lumen_mps_stream_queue`, `lumen_mps_stream_encoder`, `lumen_mps_stream_encoded`, and the `lumen_mps_completion` callback type. |
| `mps.mm` | The queue, the open command buffer/encoder, batching, counter-sample timestamps, and the `extern "C"` functions `mod.rs` calls. |

## Launch flow

1. Rust (`ops/mps/mod.rs`, `ops/fill/mps/mod.rs`) calls
   `submit(tensors, name, kernel)`: boxes a `Submission` holding the tensors
   (kept alive until the GPU is done) and, if the profiler times MPS, a
   `GpuContext` plus the paired profiler/host clocks. Returns
   `(context, completed, timed)`.
2. The ObjC shim calls `lumen_mps_stream_encoder(timed)`, which **takes the
   stream lock** and returns the shared compute encoder (a fresh one per
   timed launch, with start/end counter samples attached), encodes the
   dispatch, then calls `lumen_mps_stream_encoded(done, context)`, which
   records the op and **releases the lock**. If encoding fails before that,
   Rust calls `cancel(context)` to free the submission.
3. Every `kOpsPerCommit` (32) ops the command buffer is committed. Its
   completed handler resolves the timestamps (falling back to the buffer's
   `GPUStartTime`/`GPUEndTime`) and calls each op's `completed`.
4. `completed` converts GPU times to the profiler clock, calls
   `profiler::record_kernel_in`, drops the tensors, bumps `FINISHED` and
   notifies a condvar.

`synchronize()` flushes (commits the open buffer) and waits until
`FINISHED` reaches the number of ops encoded so far. `mark()` enqueues an
op with no kernel, so `synchronize` also waits for work encoded into the
command buffer from outside (after a plan's custom-op call, and a Core ML
step); `command_buffer()` ends the encoder and exposes the open
`MTLCommandBuffer` for external encoding (Core ML, `lumen.Stream.command_buffer`).

## Invariants & gotchas

* The encoder/encoded pair brackets a held `os_unfair_lock`; always call
  `lumen_mps_stream_encoded` after a successful `lumen_mps_stream_encoder`.
* Work is not submitted to the GPU until 32 ops accumulate or someone
  flushes/synchronizes; host reads must synchronize first.
* Counter sample buffers are pooled (at most 24 in flight); if the device
  does not support stage-boundary sampling, timings are per command buffer.
* A failed command buffer only prints a warning in `completed`.

## Related

* [`..`](../README.md) — the `Stream` trait and `lumen.Stream`.
* [`../../allocator/mps`](../../allocator/mps/README.md) — the buffers kernels bind.
* [`../../profiler`](../../profiler/README.md) — where timings go.
