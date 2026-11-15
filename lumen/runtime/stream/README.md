# runtime/stream

Device streams: a handle to lumen's work queue on each device, for host synchronization, and holds that stall a
queue. Kernel launchers (`runtime/kernel`) submit into these queues; this folder also owns the MPS command
stream, CUDA graph capture, TPU launch batches and the Python `Stream`, `Event` and `CudaGraph` classes.

## Files

- `mod.rs` — the `Stream` trait (`synchronize`), `stream_for(device)`, and `hold(device)` with its `Hold` guard.
- `cpu.rs`, `meta.rs` — `CpuStream` and `MetaStream`, whose `synchronize` does nothing (CPU work is synchronous,
  meta runs nothing).
- `cuda.rs` — `CudaStream` (the device's default stream), external-stream waits and synchronization, the CUDA
  runtime bindings (`ffi`) shared with events and NCCL, and `hold`, which parks the stream behind a host function
  until released. `CudaGraph` captures lumen's launches: `begin` redirects them to a capture stream
  (`launch_stream`), `end` instantiates the graph, `stage` copies an input in on its stream, and `replay` launches
  it again; it keeps the label of the step each node came from so a profiled replay names its kernels.
- `mps/mod.rs`, `mps/mps.h`, `mps/mps.mm` — the MPS command stream over an Objective-C++ shim: one command queue
  and one open command buffer, committed every 32 encoded ops or on `flush`. `submit` builds a completion context
  that keeps the launch's tensors alive until its command buffer finishes and, when profiling, reports its GPU
  time; `synchronize` commits and waits until every encoded buffer has completed. Also `fence`, `mark`,
  `command_buffer` and `hold` (a GPU wait on a shared event).
- `tpu.rs` — `TpuStream` (synchronizes through the TPU allocator) and the per-thread stack of open `Batch`es that
  TPU launches collect in; `Batch::run` runs a batch as one executable, `flush(index)` runs the innermost batch's
  pending launches early, and dropping a batch unrun discards them.
- `python.rs` — PyO3 classes `Stream`, `Event` and (CUDA builds) `CudaGraph`, registered into
  `lumen.runtime.stream._C`.
- `__init__.py`, `_C.pyi` — the Python package re-exporting `Event` and `Stream`, and the type stubs.
- `tests.rs` — every present device yields a stream that synchronizes; absent devices error.

## Key functions

- `stream_for(device)` — lumen's stream on an available device; errors otherwise.
- `hold(device)` — stalls the queue until the returned `Hold` drops, so work enqueued meanwhile runs back to back
  (the autotuner times with it). Only CUDA and MPS stall; other devices get an empty hold.
- `cuda::launch_stream()` — the stream CUDA kernels launch on: null (the default stream) unless a graph captures.

## Conventions

- lumen has one stream per device; a Python `Stream` with a nonzero `stream_id` wraps an external CUDA stream,
  and `wait_stream` makes it wait on lumen's stream without blocking the host.
- `command_buffer()` and `buffer(tensor)` on a Python `Stream` are MPS only, for interop with other Metal code.
- Python `synchronize` calls release the GIL while waiting.
- Memory allocated while a `CudaGraph` captures stays the graph's until it is dropped.
