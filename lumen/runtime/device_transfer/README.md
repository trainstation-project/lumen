# runtime/device_transfer

Host/device copies: the `lumen::copy_h2d` and `lumen::copy_d2h` ops, dispatched through `runtime::dispatch` to
each backend's copy kernel. Tensor moves between the CPU and a device (and staged plans moving values between
host and device stages) end up here.

## Files

- `mod.rs` — `CopyKernel`, the `COPY_H2D` / `COPY_D2H` ops with their built-in kernel tables, and the entry points
  `copy_h2d(dst, src)` and `copy_d2h(dst, src)`, which check the operands, record a profiler op and dispatch.
- `cpu.rs` — a plain `memcpy` of `dst.nbytes()`; the CPU's kernel both ways, and MPS's device-to-host kernel.
- `cuda.rs` — synchronous `cudaMemcpy` in either direction, run under the profiler's CUPTI correlation; panics on
  a CUDA error.
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
- `copy_d2h` waits for the source's pending writes (`src.wait_()`) before reading.
- MPS memory is host-visible, so reading it back is a plain `memcpy`.
- TPU device memory holds tensors in the tiled layout, never row-major, so both directions convert.
- CPU-to-CPU copies record no profiler op.
- A backend not linked into the build has no kernel for its key.
