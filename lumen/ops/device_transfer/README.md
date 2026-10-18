# `lumen/ops/device_transfer`

Host ↔ device copies of whole tensors, as two dispatcher `Op`s:
`COPY_H2D` (`lumen::copy_h2d`, dispatched on the **destination's** device)
and `COPY_D2H` (`lumen::copy_d2h`, dispatched on the **source's** device).
`Tensor` (`lumen/tensor/tensor.rs`) calls the public `copy_h2d` /
`copy_d2h` functions; they validate, profile, and dispatch to a per-backend
`CopyKernel = fn(dst: &Tensor, src: &Tensor)`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `CopyKernel`, `COPY_H2D` / `COPY_D2H` and their static registries, `copy_h2d`, `copy_d2h`, the profiler range (`transfer`) and the precondition `check`. |
| `cpu.rs` | `memcpy`: a plain `copy_nonoverlapping` of `nbytes`. |
| `cuda.rs` | `cudaMemcpy` (host-to-device / device-to-host), wrapped in CUPTI correlation; compiled only with `lumen_cuda_linked`. |
| `mps.rs` | `copy_h2d` for MPS (only with `lumen_mps_linked`). |

## Kernels per key

| Key | `copy_h2d` | `copy_d2h` |
| --- | --- | --- |
| `Cpu` | `cpu::memcpy` | `cpu::memcpy` |
| `Mps` | `mps::copy_h2d` | `cpu::memcpy` (MPS memory is host-visible) |
| `Cuda` | `cuda::copy_h2d` | `cuda::copy_d2h` |
| `Meta` | none | none |

## How MPS copies work

* **H2D** is in stream order, without a wait: the source is first copied on
  the CPU into a fresh MPS staging tensor, then a GPU copy kernel
  (`ops::dynamic_slice::mps::copy`) moves staging → `dst` on the stream, so
  the write lands after any GPU work already submitted that reads `dst`.
* **D2H** first calls `src.wait_()` (synchronizing with the stream), then
  reads the shared memory directly with `memcpy`.

## Invariants & gotchas

* `check` asserts: the host side is `Cpu(0)`; same dtype and element count;
  both contiguous; not sharing storage. Kernels rely on this and copy
  `numel · size_of` bytes.
* Empty tensors return before dispatch.
* A transfer to/from a non-CPU device is profiled as an op (`copy_h2d` /
  `copy_d2h`, src's type in, dst's out); the D2H wait is inside that range.
  CPU-to-CPU copies are not profiled.
* Writes go through `&Tensor` (shared data); callers must avoid data races.

## Related

* [`../README.md`](../README.md) — `Op`, `DispatchKey`.
* `lumen/tensor/tensor.rs` — the callers.
* `lumen/ops/wait/` — what `wait_` dispatches to.
