# runtime

What runs on devices while a program executes: op dispatch, each backend's kernel launcher, streams, events,
host/device transfers and waits. It is the last stage of the pipeline (graph_compiler -> codegen -> plan ->
runtime): plan steps and codegen call into it to launch kernels, and tensors, the autotuner, the profiler and
distributed code use its streams, events and copies.

## Files and folders

- `mod.rs` — declares the submodules; only `stream` is public outside the crate.
- `dispatch.rs` — `DispatchKey` and `Op<K>`: an op's built-in kernel per device plus kernels registered at
  runtime (see `dispatch/README.md`).
- `wait.rs` — the `lumen::wait` op (`WAIT`) and `wait(t)`, which blocks until the device holding `t` is idle:
  nothing on CPU and meta, a stream synchronize on CUDA and MPS, the allocator's wait on TPU.
- `device_transfer/` — the `copy_h2d` / `copy_d2h` ops and each backend's copy.
- `event/` — the `Event` trait and per-backend events for synchronizing and timing.
- `kernel/` — the `Launch` trait and each backend's launcher (CUDA, Triton, Metal, TPU).
- `stream/` — the `Stream` trait, queue holds, the MPS command stream, CUDA graphs, TPU batches and the Python
  `Stream`, `Event` and `CudaGraph` bindings.

## How it connects

- `plan/step/<device>.rs` runs a step's kernel, which launches through `kernel::<device>`; MPS steps use
  `kernel::mps::{capture, prepare, replay}`, TPU plans run inside a `stream::tpu::Batch`.
- `autotune` times candidates with `event::event_for` and stalls the queue with `stream::hold` so timed runs
  execute back to back.
- Backend availability is decided at build time by the `lumen_cuda_linked`, `lumen_mps_linked` and
  `lumen_tpu_linked` cfgs; a backend not linked has no module, and dispatch returns no kernel for it.

## Conventions

- Launches never wait: Metal records into the open command buffer, CUDA enqueues on lumen's stream, and a TPU
  batches into the program it runs on flush. Host synchronization goes through `Stream::synchronize`,
  `Event::synchronize` or `wait`.
- Ops are named with `op_name!`, which prefixes the library name (`lumen::wait`, `lumen::copy_h2d`).
