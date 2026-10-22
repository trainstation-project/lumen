# `lumen/kernels/mps`

The shared MPS (Metal) plumbing for graph-primitive kernels: the routing of
a plan step to its op family's `encode`, the one launch path every Metal
kernel goes through, the helpers that pack kernel arguments, the Metal
prelude every `kernels.metal` builds on, and the Objective-C++ shim that
compiles and caches pipelines and encodes dispatches.

Compiled only when `build.rs` sets `lumen_mps_linked` (macOS with Metal);
the module is `pub(crate)`. Callers:

* `graph_compiler/plan.rs` calls `encode` for every step on an MPS executor;
  `graph_compiler/mps` hands `scratch_bytes` to the planner to size each
  step's scratch.
* Each op family's `mps/mod.rs` (`elementwise`, `reduce`, `dot_general`, …)
  calls `launch` / `launch_step` and the `*_arg` helpers.
* `graph_compiler/mps` and `codegen/mps` launch generated fusion kernels
  through `launch`, and compile them with `lumen_mps_compile_kernels`.
* `custom_op/mps.rs` compiles user kernels with `lumen_mps_compile_kernel`
  and launches them via `launch`; it also owns the `Grid` type re-exported here.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `encode` (per-`Primitive` routing), `scratch_bytes`, argument packers (`u32_arg`, `u64_arg`, `dims_arg`, `dims32_arg`, `element_arg`), `BYTES_PER_THREAD`, `elementwise_grid`, `launch_step`, `launch`, `mark_event` (an `Event`'s timed marker kernel). |
| `kernels.metal` | Shared Metal definitions (no kernels): dtype predicates and ranges, saturating conversions, the binary/unary functors, Philox, `FOR_EACH_ELEMENT`, strided `offset_of`/`offset_of32`, dtype X-macros, simdgroup-fragment helpers. |
| `shim.mm` | Objective-C++: compiles the embedded primitive library once, caches pipelines by name, and exports `lumen_mps_compile_kernels`, `lumen_mps_compile_kernel`, `lumen_mps_pipeline` (a retained pipeline by name), `lumen_mps_launch_pipeline` and `lumen_mps_launch_kernel` (by name). |

## Routing (`encode`)

`encode(step, inputs, output, scratch, keep)` matches `step.primitive`:

| Primitive(s) | Goes to |
| --- | --- |
| `ReduceSum`, `ReduceMax` | `reduce::mps::encode` (with scratch) |
| `Cumsum` | `scan::mps::encode` (with scratch) |
| `Fusion` | `graph_compiler::mps::encode` (with scratch) |
| `LinearCrossEntropy` | `dot_general::mps::encode_linear_cross_entropy` (with scratch) |
| `DotGeneral` | `dot_general::mps::encode` |
| `Reshape`, `BroadcastInDim`, `Transpose`, `Slice` | `layout::mps::encode` |
| `DynamicSlice`, `DynamicUpdateSlice` | `dynamic_slice::mps::encode` |
| `ScatterAdd` | `scatter_add::mps::encode` |
| `Full`, `Iota`, `RandomBits` | `factory::mps::encode` |
| `Wait` | `wait::encode` |

`Concatenate` returns an error (it only runs inside a fusion);
`ToHost`/`ToDevice`, `FusionOutput` and `CustomCall` are `unreachable!`.
`scratch_bytes` mirrors the scratch-taking arms: it asks the family how much
workspace its partials need (0 for everything else).

## Launching

`launch(kernel, buffers, args, grid, keep, name)`:

* `buffers` are device pointers bound at indices `0..n`; each `args` entry
  is raw bytes bound with `setBytes` at `n + j`. `launch_step` is the common
  case: the step's inputs, then its output.
* `Grid::Threads([x, y, z])` dispatches that many threads, threadgroups of
  up to 256 threads (capped by the pipeline's max) filling x first.
  `Grid::Groups([x, y, z])` dispatches that many threadgroups of
  **16 × 16** threads; kernels written for it assume 256 threads
  (`tid.y * 16 + tid.x`).
* A grid with a zero dimension launches nothing (and clears the profiler's
  pending launch).
* A kernel's pipeline is looked up by name once (`lumen_mps_pipeline`,
  retained) and kept in `launch`'s own map; later launches pass the
  pipeline itself (`lumen_mps_launch_pipeline`), with no C string, lock or
  allocation on the way. At most 31 buffers and arguments together, Metal's
  limit.
* `keep` holds the tensors (and workspace) alive until the GPU finishes;
  `stream::mps::submit` turns it into the completion context. A plan keeps
  each tensor's storage only (`Tensor::holding`: no shape or strides copied).
* A non-zero shim status (unknown kernel name, a pointer that is not an MPS
  buffer, a grid dimension over `u32::MAX`) becomes `Err`.

`elementwise_grid(n, dtype)` gives each thread `n / 32K` elements of output,
at least one and at most 16 bytes' worth (`BYTES_PER_THREAD`): a small output
runs on as many threads as the GPU holds at once (an M4 Pro's 20 cores, about
20K), since there latency, not bandwidth, sets its time; a large one on 16
bytes a thread, for bandwidth. Kernels loop with `FOR_EACH_ELEMENT`, which
strides by `threads_per_grid` up to `per_thread<T>()` times and stops at `n`,
so any grid of at least `n / per_thread<T>()` threads covers the output.

## The Metal library

`build.rs` concatenates `PRIMITIVE_KERNELS` — this folder's `kernels.metal`
first, then elementwise, reduce, scan, dot_general, layout, dynamic_slice,
scatter_add, factory — into one source embedded as `kPrimitiveKernels`
(`primitive_kernels.h` in `OUT_DIR`). The shim compiles it on first use with
`fastMathEnabled = NO`. So every family's `.metal` may use anything defined
earlier, and kernel names must be unique across all files.

Kernel names encode the dtype by its lumen name (`u8`…`i64`, `f16`, `bf16`,
`f32`, `bool`), via the `FOR_NUMERIC` / `FOR_ALL` / `FOR_FLOAT` X-macros;
mixed-precision variants append the accumulator (e.g. `_f16_f32`).
`attention/mps/*.metal` is not part of this library: fusion kernels include
it in their prelude (`graph_compiler/mps/mod.rs`, with `TEMPLATES_ONLY`
defined so only templates, not the primitive kernels, are emitted).

## Invariants & gotchas

* Conversions follow the reference: float → int saturates and maps NaN to
  0; integer division by 0 gives -1; `max_op` propagates NaN.
* `encode` errors if the output has more than `u32::MAX` elements.
* `dims_arg`/`dims32_arg` pass a single `0` for an empty list, so a
  rank-0 argument is never zero bytes.
* Pipelines are cached by name and never evicted; `lumen_mps_compile_kernels`
  skips function names already in the cache.

## Related

* [`../README.md`](../README.md) — the op layer and dispatch.
* `lumen/stream/mps/` — the command stream (`lumen_mps_stream_encoder`, `submit`).
* `lumen/allocator/mps/` — `lumen_mps_buffer`, mapping a pointer to its `MTLBuffer`.
* `lumen/custom_op/mps.rs` — `Grid` and user Metal kernels.
* `build.rs` — `PRIMITIVE_KERNELS`, `embed_metal_source`.
