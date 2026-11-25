# graph_compiler/tests

Rust unit tests of the graph compiler (`#[cfg(test)] mod tests` in `graph_compiler/mod.rs`). They build
small graphs by hand, run single passes or whole compiles, and check the rewritten graph's structure and,
where values matter, its results against the reference interpreter (`crate::reference`).

## Files

- `mod.rs` - device-independent tests:
  - CSE (repeated computations run once; different parameters or operand order are kept).
  - `Simplify` (algebraic identities, transposes of dots swapping operands, rounding casts folding into
    dots, rearranges and casts moving past elementwise and layout ops).
  - Host stages (host values run in a host stage, programs without host values compile to one plan,
    mixing host and device operands is an error).
  - `pipeline::run` running passes in order and dumping the graph after each one.
  - Attention operand strides, `linear_reordering` in every operand layout, and `preserve_precision`.
  - `devices_run_the_passes_they_implement`: the exact pass list each device's pipeline runs under the
    default config (CPU, CUDA, Triton, TPU). Update it when a pass is added, removed or reordered.
- `mps.rs` - MPS pass and codegen tests, compiled only when MPS is linked (`lumen_mps_linked`): fusion of
  elementwise and layout chains, dot merging and packed parameters, dot canonicalization, slices read in
  place, reduction and GEMM epilogues, multi-output fusion, RMS norm and softmax row kernels, split-K and
  every GEMM/GEMV kernel candidate. Device checks run only when a GPU is available.
- `tpu.rs` - TPU pass tests: normalization rows (`diamonds::diamonds` then `diamonds::fused`) fusing when
  values other kernels read from them are written out.

## Helpers

- `simplified(g, matched)` runs `Simplify` alone through `pipeline::rewritten`, which applies one pass to
  a plain module with default options.
- `apply`, `ty`, `names`, `full` and `matmul` build graphs and list primitive names.
- MPS tests reuse `plan::tests::mps` (`available`, `check`, `values`) and an MPS `merge_dots::Target`
  matching the MPS pipeline's.
