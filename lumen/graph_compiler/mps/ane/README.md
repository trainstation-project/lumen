# `lumen/graph_compiler/mps/ane`

Optional Apple Neural Engine offload for MPS plans, through Core ML (the
only public way onto the ANE). Gated by `lumen.config.compiler.neural_engine`
(off by default: the ANE accumulates float16 dots wider than float16 but
narrower than float32, so results differ from lumen's kernels).

The `NeuralEngine` pass runs inside the MPS pipeline (after `simplify`,
before `dot_strength` / `split_k`) and replaces regions of float16 work on
fixed weights with single `Primitive::Fusion` steps named `coreml_*`
(`crate::graph::NEURAL_ENGINE`). At run time `mps::encode` hands those steps
to `encode` here, which runs a Core ML prediction directly on the plan's
Metal buffers, synchronized with the GPU stream by a shared event.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `NeuralEngine` pass, `offload` (region finding and rewrite), `Program` (compiled Core ML model), the `PROGRAMS` registry, `encode`, the prediction thread, and `invalidate`. |
| `coreml.mm` | Objective-C++ shim (`lumen_coreml_*`): compile an ML Program spec at runtime, report per-op devices via MLComputePlan, predict over raw buffers, and signal/wait on a shared `MTLSharedEvent`. Built by `build.rs`. |

## Offload (compile time)

- **Fixed weights**: the pass marks inputs that are parameters not written
  by the program (not in `donate` / `donate_into`), and packed blocks none
  of whose members are written. Only these are baked into the program.
- **Seeds**: an all-float16 dot with exactly one operand computed from
  fixed weights and at least `MIN_MACS` (2^24) multiply-adds. Dots with a
  float32 result are never offloaded.
- **Growth**: the region grows in node order over float16 work reading its
  values (bool/int32 masks and indices allowed) that `lowerable` accepts,
  keeping it convex so it is one step. Other dots join only if float16 and
  reading a fixed weight.
- **Placement check**: each region is lowered to MIL
  (`crate::codegen::mps::ane::mil`) and compiled with zeroed constants;
  whatever Core ML would run off the ANE (on the CPU), plus what depends on
  it, is dropped from the region, up to 4 tries. If Core ML refuses it, the
  seed itself is placed off the ANE, or it does not settle, nothing of it
  is offloaded.
- **Rewrite**: each region becomes a `coreml_*` fusion; its `Entry` (body,
  which inputs are constants, program = `None`) is stored in `PROGRAMS`.

## Run time (`encode`)

1. First run: `stream::mps::synchronize()`, read the weights' current
   values on the host, compile the program with them baked in.
2. Unless the previous Core ML step directly preceded this one, encode a
   GPU signal on the open command buffer and flush.
3. Send a `Job` to the single `lumen-coreml` thread: it waits for `ready`,
   predicts in place on the plan's (unified-memory) buffers, then signals
   `done` — even on failure, which is only printed.
4. Encode a GPU wait on `done` and `stream::mps::mark()`, so later kernels
   and `synchronize` wait for the prediction.

`invalidate(steps)` (Python: `Plan._invalidate_neural_engine`, called by
`lumen.compile` when a weight's `_version` changes) drops compiled programs
so they rebuild from the new weights on their next run.

## Invariants & gotchas

- The plan never reuses buffers in place across a `coreml_*` step
  (`plan.rs`): Core ML reads and writes them in no stated order.
- Predictions run one at a time, in step order, on one thread; `Program`
  and `Job` are `Send`/`Sync` by `SAFETY` arguments that rely on this and on
  `keep` holding the buffers alive.
- Weights are baked as constants: any write to a fixed weight without
  `invalidate` leaves the program stale.
- The pass runs before `split_k`, so it sees whole dots.

## Related

- [`..`](../README.md) — the MPS pipeline and `encode` dispatch to this folder.
- [`../../`](../../README.md) — `CompilerConfig::neural_engine`, the planner.
- [`../../../codegen/`](../../../codegen/) — `mps/ane`: MIL lowering and the Core ML protobuf.
- [`../../../stream/`](../../../stream/) — the MPS stream whose command buffers carry the signal/wait.
