# graph_compiler/pipeline

Running passes, and each device's list of them.

## Files

- `mod.rs` - `Pipeline`, a named, ordered list of passes: `add` appends a built pass, `maybe` one that may be off
  (`None`), `pass::<P>` pass `P` as `CompilerPass::configured` gives it under a config, `pass_if::<P>` the same under a
  condition of the device's own. `run` runs each pass in a profiler range `<pipeline>::<pass>`, dumping the graph
  after each with `dump_to`. `Passes` is a device's pass list; `GraphModule` the module tests run single passes on;
  `rewritten` / `rewritten_on` run one pass in tests.
- `cpu.rs`, `cuda.rs`, `mps.rs`, `triton.rs`, `tpu.rs` - each device's `Passes`: the passes it runs, in order, the
  values it gives them (merged dots' `Target`, update chain lengths, the GEMM row tile) and the passes it runs only
  with `fuse` (every GPU's fusing passes; TPU runs all of its always). `mps.rs` builds only when MPS is linked.

## Invariants

- Every device keeps the passes it shares in the order of `passes/README.md`; Triton makes MPS's choices
  (its list is MPS's minus the Neural Engine, `canonicalize_dots` and `concats`, plus `wide_rows` and
  `horizontal_fusion`).
- `graph_compiler/tests` checks each device's default list (`devices_run_the_passes_they_list`).
