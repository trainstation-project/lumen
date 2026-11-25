# passes/apple_neural_engine

The Apple Neural Engine pass. On MPS it moves large float16 GEMM regions that read fixed weights out of the
Metal graph into Core ML programs, which run on the Neural Engine. It runs in MPS's pass list after
`simplify` and the late dot merges, before `linear_reordering`, `split_k` and fusion.

## Files

- `mod.rs` - re-exports `mps::NeuralEngine`, which only MPS's pass list runs. Its submodules are compiled only when
  MPS is linked.
- `mps.rs` - the pass, the program registry and the runtime:
  - `NeuralEngine::apply` marks inputs as fixed when they are parameters (or packed blocks) that are never
    donated, then calls `offload`.
  - `offload` seeds a region at each float16 `DotGeneral` where exactly one operand is computed from
    fixed weights and the output size times K is at least `MIN_MACS`. It grows the region forward through
    readers that MIL can lower, keeping the region convex. It compiles the region with zeroed weights to
    see where Core ML places each op, and drops ops kept off the ANE (up to four tries).
  - `rewrite` replaces each region with a `Fusion` named `coreml_<n>` (`graph::NEURAL_ENGINE`) and
    `FusionOutput`s for its other outputs.
  - `encode` (called from `runtime/kernel/mps.rs`) queues a step on a host Core ML worker thread,
    compiling the program with the real weights on first use. The Metal stream signals a shared event
    before the job and waits for it after.
  - `invalidate` forgets compiled programs so they recompile with new weight values
    (`Plan._invalidate_neural_engine`).
- `mil.rs` - `lower`: turns a region body into a serialized Core ML `mlProgram` spec in the MIL dialect,
  embedding fixed inputs as constants and exposing the rest as `in{k}` features. Also maps dtypes
  (MIL computes float16, float32, int32 and bool).
- `proto.rs` - a minimal protobuf wire-format writer (`Message`) used by `mil.rs`.
- `coreml.mm` - the Objective-C shim (built by `build.rs`): compile, predict on strided inputs, report
  each op's device, and signal and wait on the shared Metal event.

## Config

- `neural_engine` (off by default) turns the pass on.
- `Plan.neural_engine` reports whether a compiled plan runs any Neural Engine steps.
