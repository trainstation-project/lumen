# `lumen/profiler`

A profiler modeled on PyTorch's (`torch.profiler`, `torch/csrc/profiler`).
While a session runs it records tensor ops, user ranges, plan steps, memory
traffic and device-side kernel timings; when it stops it hands back a
`Profile` that can be listed as raw events, aggregated into a PyTorch-style
`key_averages()` table, or exported as a Chrome trace (Perfetto /
`chrome://tracing`).

The rest of lumen calls in through cheap hooks: `record_op` (tensor ops and
compiled plan steps), `report_memory` (allocators), `gpu_context` /
`record_kernel_in` (the MPS stream), and `cupti::correlated` (CUDA launches).
When no session is running, every hook is one relaxed atomic load.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Session state (`start` / `stop`), `ProfilerConfig`, `Activity`, `Event` / `EventKind`, `RecordGuard`, and all hooks the rest of the crate calls. |
| `summary.rs` | `Profile`, `EventAvg`, `SortBy`: per-name aggregation (`key_averages`), the summary `table`, and PyTorch-style time/memory formatting. |
| `chrome.rs` | Chrome trace JSON writer (`trace`), with PyTorch's categories and arg keys. |
| `cupti.rs` | CUDA device timing through CUPTI's activity API; compiled only with `cfg(lumen_cupti_linked)`. |
| `tpu.rs` | TPU device timing through libtpu's own profiler (PJRT's profiler extension), its XSpace read by a small protobuf reader; compiled only with `cfg(lumen_tpu_linked)`. |
| `python.rs` | `lumen._C` bindings: `_profiler_start/_stop/_enabled/_cuda_timing`, `_record_function_enter`, `_RecordFunction`, `_Profile`. |
| `__init__.py` | The `lumen.profiler` Python API (see below). |
| `_C.pyi` | Type stubs for `python.rs`. |
| `tests.rs` | Rust tests: session exclusivity, nesting, shapes, memory attribution, table/sort/format, Chrome layout, Metal and CUDA timing (hardware-gated). |

## Sessions

* `start(ProfilerConfig)` errors if a session is already running — one
  session per process. It turns the config into bit flags in `FLAGS`
  (`CPU`, `MEMORY`, `CUDA`, `MPS`, `SHAPES`) and bumps `SESSION`.
* `ProfilerConfig { activities, profile_memory, record_shapes }`; default is
  CPU only, no memory, no shapes. `Activity` is `Cpu`, `Cuda`, `Mps` or `Tpu`.
* `stop()` first synchronizes the MPS stream (and, via `cupti::stop`, every
  CUDA device) so in-flight GPU work lands, clears the flags, sorts events by
  `(start_ns, id)`, then copies each op's types/strides/accum onto the GPU and
  host-kernel events it issued — unless the launch recorded its own types.
* Events are tagged with the session id when opened; `push` drops events
  from an earlier session. Times are nanoseconds relative to session start
  (`now_ns` is a process-wide monotonic clock).

## Event kinds

| `EventKind` | Python `kind` | Recorded by | Chrome `cat` |
| --- | --- | --- | --- |
| `Op` | `op` | `record_op` (tensor ops, plan steps, `synchronize`, transfers) | `cpu_op` |
| `UserRange` | `user_range` | `record_function` | `user_annotation` |
| `Memory` | `memory` | `report_memory` from `allocator/cpu.rs` and `Region` | `cpu_instant_event` (`[memory]`) |
| `Gpu` | `gpu` | `record_kernel_in` / `record_gpu_in` (MPS completion handler, CUPTI) | `kernel`, `gpu_memset`, `gpu_memcpy` (by name prefix) |
| `HostKernel` | `host_kernel` | `record_host_kernel` (a plan step run on the host), `record_host_kernel_in` (Core ML) | `kernel` |

Ops and ranges nest per thread through a thread-local id stack: each
`RecordGuard` records its `parent`, and dropping it (even out of order) pops
it. Device work captures a `GpuContext` (session, parent op, thread, and an
optional `Launch`) at submission time, so its event — which arrives later
from a completion handler — is still parented to the op that issued it.
`next_launch(name, types)` lets an op that launches several kernels (split
reductions, `linear_cross_entropy`) name each launch and give it its own
types; `clear_next_launch` drops one that did not happen.

## `record_shapes`

With `record_shapes`, ops record (via closures that only run then):

* `inputs` / `outputs`: `TensorType` (dtype + shape) of each.
* `input_strides` / `output_strides`: per tensor, the strides in elements
  as the kernel actually reads/writes it. Plan steps
  (`graph_compiler/plan.rs`) fill these from `Step::input_strides` /
  `output_strides`: a view's strides (a slice read in place, an in-place
  concatenate's operand written into its result), else contiguous strides.
  Ops that do not call `RecordGuard::strides` have none.
* `accum`: accumulation dtypes (`Primitive::accum_dtypes`: a dot's or sum's
  `accum_dtype`, a fusion's per-reduction dtypes; empty for elementwise).

## Device timing

* **MPS** (`Activity::Mps`): `stream/mps` captures a `GpuContext` and the
  host clock in `submit`; the Metal stream samples per-dispatch GPU
  timestamps (counter sample buffers, falling back to the command buffer's
  `GPUStartTime`/`GPUEndTime`), and the completion handler converts them to
  the profiler clock and calls `record_kernel_in` with the Metal kernel name.
* **CUDA** (`Activity::Cuda`): `cupti.rs` enables CUPTI activity kinds
  (kernel, memcpy, memset, runtime, driver, external correlation).
  `correlated(device, work)` pushes an external correlation id mapped to the
  current `GpuContext`; at `stop`, records are flushed and mapped back to their
  ops, with timestamps shifted by a (CUPTI ns, profiler ns) pair taken at
  `start`. Without CUPTI linked, starting a CUDA session from Python raises
  (`CUDA_TIMING_MISSING` tells you to set `CUPTI_LIB_DIR`).
* **TPU** (`Activity::Tpu`): `tpu.rs` starts libtpu's profiler (once the
  programs in flight have run) and `ops/tpu`'s `launch` records each
  kernel's `GpuContext` and name (the variant its spec makes, as MPS names
  its kernels: `matmul_bf16_f32_bf16_2048x1024x4096`, `max_bf16_512x1024`).
  At `stop`, libtpu's
  XSpace gives each device's kernel runs in order (plane `/device:TPU:<i>`,
  line `XLA Ops`, its custom calls: a plan's kernels are one program, each a
  `tpu_custom_call`), and a device runs its kernels in the order they were
  launched: the n-th run is the n-th launch's, named as the op (plan step)
  that launched it. libtpu's clock is its own, so the
  runs are shifted onto the profiler clock by causality: the least shift
  that starts no run before its launch (each launch records when its
  program was launched), which starts the most launch-bound run at its launch, to within
  the device's launch latency. Without the TPU backend, starting a TPU
  session errors.

## `key_averages` and the table

`Profile::key_averages` groups by name in first-seen order. Memory events
get no row; they count toward their enclosing ops' `cpu_memory_usage` /
`device_memory_usage` (total = inside, self = directly). GPU events get rows of
their own (`is_device_event`), except one named as the op that issued it (an
MPS or TPU plan step's kernel run, its kernel in `kernel`): that is the op's
row's device time alone, not a second row of the same name counting it
again. Ops get `device_time_total` from work issued inside them. `table(sort_by, row_limit)` sorts descending by a `SortBy`
(PyTorch names; `cuda_*` aliases map to the device columns), labels the device
columns `CUDA`, `MPS`, `TPU` or `DEVICE`, and only shows device/memory columns
when there is something to show. The self-device total sums the GPU events
once, not again in the ops that issued them.

## Chrome trace

`chrome::trace` emits `"X"` events per op/range/kernel plus `ac2g` flow arrows
from the issuing op to its GPU or host-kernel work. Arg keys follow PyTorch:
`External id`, `Input Dims`, `Input type`, `Input Strides`, `Output Dims`,
`Output type`, `Output Strides`, plus `Accum type`; GPU events add `device`,
`stream` (7), `correlation`, and `kernel`; memory events carry `Device Type`,
`Device Id`, `Addr`, `Bytes`, `Total Allocated`, `Total Reserved`. Processes:
"CPU dispatch" (ops), one per device ("CUDA i" / "MPS"), "Neural Engine" (host
kernels named `coreml*`), and "CPU" (host kernels). `record_function` ranges
are also drawn as `gpu_user_annotation` spans covering their device work.

## Python API (`__init__.py`)

* `ProfilerActivity` (`CPU`, `CUDA`, `MPS`, `TPU`) and `supported_activities()` —
  CPU plus each device that can allocate (CUDA also needs CUPTI).
* `profile(activities=None, profile_memory=False, record_shapes=False)` —
  a `with` block (or `start()` / `stop()`); defaults to all supported
  activities. Afterwards: `events()` (list of dicts), `key_averages()`
  (`EventList` of `FunctionEventAvg`, with `.table(sort_by, row_limit)`;
  times in µs), `export_chrome_trace(path)`.
* `record_function(name)` — context manager or decorator. It also opens a
  tracer scope (`lumen.graph.tracer._enter_scope`), so inside
  `lumen.compile` the traced ops' plan steps run inside a range of that name.

## Gotchas

* Strides are recorded only by `RecordGuard::strides` (called for
  compiled plan steps) and by `next_launch`'s `IoTypes`; eager tensor ops
  have none.
* A GPU event inherits its op's types at `stop`, not at record time.
* CPU frees are reported only for blocks allocated while profiling memory
  (see `lumen/allocator/README.md`).
* `Profile::for_thread` filters events to one thread (used by tests to
  ignore concurrent activity).

## Related

* [`../allocator`](../allocator/README.md) — memory events.
* [`../stream/mps`](../stream/mps/README.md) — Metal timestamps and completion.
* `lumen/graph_compiler/plan.rs` — plan steps recorded as ops and host kernels.
* `tests/test_profiler.py` — Python-side tests.
