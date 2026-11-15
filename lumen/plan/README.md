# plan

A plan is a compiled program ready to run: an ordered list of kernel steps whose values live in caller inputs,
freshly allocated outputs or one shared workspace. It sits between codegen and the runtime: `graph_compiler`
lowers a graph to a `CpuPlan`, each device's codegen maps that plan's steps into its own step type, and running a
plan launches each step through `runtime`.

## Files

- `mod.rs` — the plan types: `PlanOptions`, `DeviceLayout`, `PlanBase`, the `Plan` trait, `DevicePlan<S>` and its
  per-device aliases (`CpuPlan`, `CudaPlan`, `TritonPlan`, `MpsPlan`, `TpuPlan`), and `StagedPlan`. Also the
  plan listing (`Display`).
- `compile.rs` — `CpuPlan::compile` / `compile_with`: places every value and builds the steps; `StagedPlan::new`.
- `run.rs` — running plans: `Buffers` (the tensors a step resolves its `Buffer`s against), `Plan` for
  `DevicePlan` and `StagedPlan`, custom-op calls, and profiler scope ranges.
- `step/` — the step types each device runs (see `step/README.md`).
- `tests.rs` — placement tests (buffer reuse, donation, scheduling, scratch, views) and, with MPS linked, runs
  against the reference evaluator.

## Key types

- `PlanBase` — what every plan shares: input and output types, where each input is read from (`inputs_at`) and
  each output written (`outputs_at`), output aliases into inputs, workspace size and alignment, and packed
  parameter blocks. `DevicePlan` and `StagedPlan` deref to it.
- `Plan` — `steps`, `run_on(inputs, device)` (allocates workspace and outputs), `run_in(workspace, inputs)` (an
  owned plan in a caller workspace), and `run` (on the first input's device).
- `DevicePlan<S>` — one device's steps; `map` turns it into another device's plan step by step, which is how
  codegen lowers a `CpuPlan`.
- `StagedPlan` — host and device `Stage`s, each with its own plan, its workspace at an offset of the parent's, and
  `Source`s saying where its inputs come from (a program input, or a stage output, possibly copied across).
- `PlanOptions` — scratch sizing, donation, parameters (owned plans), runtime scalars, a memory limit, packed
  blocks and a device layout.

## Compiling

`compile_with` tries the traced order plus the list and DFS schedules from `graph_compiler::schedule`, keeps the
one with the smallest workspace, and calls `graph_compiler::remat` when that still exceeds `memory_limit`. For one
order it aliases reshapes, waits and in-place writes onto their operand's buffer, writes operands of in-place
concatenates straight into the result's buffer, reads view slices at an offset, places donated outputs in their
donor, and packs the remaining values and kernel scratch into the workspace by live range (largest first, lowest
non-overlapping aligned offset).

## Invariants

- Every workspace value and scratch region is aligned to `ALIGNMENT` (256 bytes), or to `DeviceLayout::align`
  when the device stores tensors in its own layout (TPU tiles).
- An owned plan (`parameters` set) has no `Output` buffers; its workspace outputs come back as views of the
  workspace, so the next run overwrites them.
- `run_on` makes inputs contiguous. `run_in` passes a non-contiguous parameter as given when every step reading it
  accepts its strides (`Step::reads_strided`); `execute` then gives those steps a `View` of the input's strides
  for that run (`Buffers::viewed`).
- CPU and TPU plans reject steps that read an operand through a view compiled into the plan (a slice read in
  place); see `step::viewed`.
- A plan runs on the meta device by returning empty outputs, without launching anything.
- `Step::batched` wraps the whole step loop, so a device that batches launches (TPU) runs them as one program.
