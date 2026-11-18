# runtime/device_transfer

Host/device copies: the `lumen::copy_h2d` and `lumen::copy_d2h` ops, dispatched through `runtime::dispatch` to
each backend's copy kernel. Tensor moves between the CPU and a device (and staged plans moving values between
host and device stages) end up here.

## Files

- `mod.rs` — `CopyKernel`, the `COPY_H2D` / `COPY_D2H` ops with their built-in kernel tables, and the entry points
  `copy_h2d(dst, src)` and `copy_d2h(dst, src)`, which check the operands, record a profiler op and dispatch.
- `pending.rs` — asynchronous copies to the host for staged plans: `to_host(src)` starts the copy and returns the
  host tensor at once, recording it by address; `wait(addr)` (a host stage's `device_wait` step) blocks for it and
  `wait_all` for what is left when the run ends. On CUDA the copy is a `cudaMemcpyAsync` into the pinned pool with
  an event to wait on; on other devices the copy runs at the wait.
- `cpu.rs` — a plain `memcpy` of `dst.nbytes()`; the CPU's kernel both ways, and MPS's device-to-host kernel.
- `cuda.rs` — `cudaMemcpyAsync` on the default stream, run under the profiler's CUPTI correlation, so a copy runs
  after the device work queued before it: host-to-device returns without waiting for the device (the driver has
  staged the source), device-to-host waits for the stream so the host can read the result; panics on a CUDA error.
- `mps.rs` — host-to-device only: copies the host data into a fresh MPS staging tensor, then copies that into
  `dst` with a GPU copy kernel, so the write is ordered after work already queued on `dst`.
- `tpu/mod.rs` — host-to-device tiles the host bytes into the TPU tiled layout and launches a DMA kernel through
  the TPU launcher; device-to-host first flushes the device's pending launches, reads the tiles back through the
  TPU allocator and untiles them.
- `tpu/__init__.py` — the Pallas program (`program(spec)`) the TPU host-to-device launch lowers, loaded as
  module `lumen.runtime.device_transfer.tpu`: one async copy of the host rows into the destination rows.

## Invariants

- The host side of a copy is a CPU tensor; both tensors are contiguous, distinct storages, and alike in dtype and
  element count (checked with asserts in `check`).
- Empty copies return before dispatch.
- `copy_d2h` waits for the source's pending writes (`src.wait_()`) before reading. A staged plan's copies to the
  host go through `pending` instead: nothing reads their result before its wait.
- MPS memory is host-visible, so reading it back is a plain `memcpy`.
- TPU device memory holds tensors in the tiled layout, never row-major, so both directions convert.
- CPU-to-CPU copies record no profiler op.
- A backend not linked into the build has no kernel for its key.
