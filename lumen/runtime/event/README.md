# runtime/event

Device events: a mark placed after the work queued so far on a device, which the host can query, wait on, and time
against another event. The autotuner times candidates with them, the compiler's Python side times runs with them,
and `lumen.Event` (`runtime/stream/python.rs`) wraps them for Python.

## Files

- `mod.rs` — the `Event` trait, `event_for(device, timing)`, `Recorded` and the host clock `host_seconds`.
- `cpu.rs` — `CpuEvent`: CPU work runs synchronously, so recording marks the current host time as already done.
- `mps.rs` — `MpsEvent`: recording dispatches an empty `event_marker` kernel (`runtime::kernel::mps::mark_event`)
  whose command buffer's completion sets the marks; `synchronize` first flushes the MPS stream so that buffer runs.
- `cuda.rs` — `CudaEvent`: a `cudaEvent_t` (created on first record, without timing unless asked) recorded on the
  default stream; `query`, `synchronize` and `elapsed_time` go straight to the CUDA runtime.
- `tpu.rs` — `TpuEvent`: recording flushes the device's pending batch and hooks a callback on the PJRT ready event
  of its latest work; with no work pending it completes immediately.

## Key types

- `Event` — `device`, `timing`, `record`, `marks`, plus default `recorded`, `query`, `synchronize` and
  `elapsed_time(end)` (milliseconds) built on `marks`. CUDA overrides them, since its times come from the device.
- `Recorded` — the completion of one record: start and end seconds and whether the work succeeded, set once by the
  backend's callback and waited on with a condvar. `Recorded::now()` is an already-completed mark.
- `event_for(device, timing)` — an unrecorded event on an available device; errors on the meta device.

## Invariants

- An unrecorded event counts as finished: `query` is true and `synchronize` returns at once.
- `elapsed_time` needs both events on the same device, both created with timing, and both recorded.
- CPU and TPU marks are `host_seconds` at completion; MPS marks are the GPU start and end times of the marker's
  command buffer. The default `elapsed_time` measures from the start event's end time to the end event's start
  time.
- `synchronize` errors if the work before the event failed.
