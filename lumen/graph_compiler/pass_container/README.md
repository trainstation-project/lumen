# graph_compiler/pass_container

Running passes, and each device's list of them.

## Files

- `mod.rs` - `Passes`, a device's passes: `passes(config)` is its list, each pass pushed in order, inside an `if`
  when it runs only under some config flags (its own, such as `flash_attention`, and the device's, such as `fuse`);
  `run(module)` runs the list in order, each pass in a profiler range `<device>::<pass>`, dumping the graph after
  each with `dump_to`. `GraphModule` is the module tests run single
  passes on; `rewritten` / `rewritten_on` run one pass in tests.
- `cpu.rs`, `cuda.rs`, `mps.rs`, `triton.rs`, `tpu.rs` - each device's `Passes`: the passes it runs, in order, the
  values it gives them (merged dots' `Target`, update chain lengths, the GEMM row tile) and the passes it runs only
  with `fuse` (every GPU's fusing passes; TPU runs all of its always). `mps.rs` builds only when MPS is linked.

## Invariants

- Every device keeps the passes it shares in the order of `passes/README.md`; Triton makes MPS's choices
  (its list is MPS's minus the Neural Engine, `canonicalize_dots` and `concats`, plus `wide_rows` and
  `horizontal_fusion`).
- `graph_compiler/tests` checks each device's default list (`devices_run_the_passes_they_list`).
