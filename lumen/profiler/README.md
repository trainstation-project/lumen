# profiler

The profiler: one global session that records host op ranges, user ranges, memory events and device kernel timings as nested events, then summarizes them PyTorch-style or exports a Chrome trace. The Python API mirrors `torch.profiler`.

## Files

- `mod.rs` holds the session (`start`, `stop`, `is_enabled`), `Activity` (CPU, CUDA, MPS, TPU), `ProfilerConfig`, `Event` and `EventKind` (op, user range, memory, device kernel, plan step), and the hooks the rest of the crate calls: `record_function` and `record_op` (returning a `RecordGuard` that closes the range on drop), `next_launch` (names and types the next device launch on this thread), `gpu_context` and `record_kernel_in` (for device timings), `untraced` (hide work from the profiler, e.g. autotuning) and `now_ns`.
- `summary.rs` holds `Profile`, the finished session, with `key_averages` and `table`.
- `chrome.rs` exports events as a Chrome trace (Perfetto or `chrome://tracing`) in PyTorch's trace layout.
- `cupti.rs` times CUDA kernels and copies through CUPTI's activity API and correlates them to the op that launched them (`correlated`, `replayed` for CUDA graph replays: each replayed kernel takes the label and input and output types of the step that launched it while the graph captured).
- `tpu.rs` records TPU launches during a session, then matches them to the custom-call runs in the device profile (an XSpace protobuf) at `stop`.
- `launched.rs` keeps a per-thread record of kernels launched while capturing, with their generated source; `lumen.viz.dump_code` and the graph compiler's dump helpers read it.
- `python.rs` binds the native session, `_Profile` and `_RecordFunction`.
- `__init__.py` is the Python API: `profile` (context manager with `events`, `key_averages`, `export_chrome_trace`), `record_function` (context manager and decorator), `ProfilerActivity`, `supported_activities`, `EventList`, `FunctionEventAvg`.
- `_C.pyi` types the native module; `tests.rs` holds unit tests.

## How it connects

- The plan runner (`lumen/plan/run.rs`) opens a range per traced scope and an op event per step.
- Kernel launchers and streams in `lumen/runtime` report device launches: the MPS stream times command buffers and records them with `record_kernel_in`; CUDA goes through CUPTI; TPU through the device profile.
- Allocators (`lumen/allocator`) report allocations and frees as memory events when memory profiling is on.

## Conventions

- Event times are nanoseconds since the session started; children point at their enclosing range with `parent`.
- Only one session runs at a time.
- Timing CUDA needs a build with CUPTI; `supported_activities()` leaves out any device that is not usable in this build and machine.
- Work wrapped in `untraced` on a thread is invisible to the session; the autotuner times candidates this way.
