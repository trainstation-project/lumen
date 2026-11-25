# fusion

The `fusion` pass groups nodes into `Fusion` nodes, each compiled by the backend to one kernel:
elementwise chains, reductions with their prologues and epilogues, normalization rows found by the
`diamonds` pass, and GEMMs with their elementwise epilogues. It runs late, after `place_concats` and
before `wide_rows`, `place_views`, `alone`, `concats` and `horizontal`; it consumes the module's `rows`
(set by `diamonds`) and clears them.

## Files

- `mod.rs` — the `Backend` description, the shared driver `fuse(graph, rows, scalars, backend, kernel)`,
  the `Fuses` trait (a device's fusion) and `Fusion`, the pass running it.
- `analysis.rs` — `Analysis`: producers, users, liveness, which nodes are fusible, and helpers
  (`depends`, `last_read`, `recomputed`) the stages share.
- `rows.rs` — `extended`: grows each normalization row forward through its first elementwise reader while
  it stays one kernel, keeping the old root as an output when others still read it.
- `reduction_epilogues.rs` — for each fusible reduction outside a row and not already hosting a GEMM's
  epilogue (a split-K combine, from `dot_epilogues`, which runs first), the longest following chain of
  elementwise ops and reshapes nobody else reads, run in the reduction's kernel.
- `dot_epilogues.rs` — for each float GEMM, the longest elementwise chain that can run in its epilogue:
  split-column regions when readers slice the output into column regions, auxiliary outputs (at most
  `max_aux`, a value read only by a narrowing cast written already narrowed), paired GEMMs, and outputs
  expanded into a concatenation's slots (`expansion`). Epilogues only read values finished before their dot.
  With `combine_epilogues`, a split-K combine (a sum over axis 0 of a batched dot's products only it reads)
  hosts the same epilogue its GEMM would unsplit, column regions included: its value is the sum through the
  cast rounding it (`regions::combine_value`, as a dot's output dtype), and that cast is one of the host's
  own nodes. No pairs or expansions there. Its value may also be read whole besides its regions (prefill
  attention reading a merged projection's values as a view of the product): the cast is then one more end,
  so the epilogue stores the value too. With `update_epilogues`, an epilogue (a region's, or a chain's) may
  end in a `dynamic_update_slice` writing its value into another operand's slot (a KV cache at `pos`): the
  update joins the fusion (`DotEpilogues::updates`, made fusible after the fact; any other update stays a
  kernel of its own), and counts at its update's size.
- `hosts.rs` — `roots`: the nodes that end a fusion (unfusible nodes, reductions outside epilogues, epilogue
  ends, row roots, outputs, values read by unfusible nodes, expensive ops read more than once); `hosts`:
  a later fusion that can also write a root's value as an extra output (multi-output fusion), plus GEMM
  auxiliary outputs and row outputs and partials hosted by their row.
- `emit.rs` — gathers each root's members, splitting a fusion that reads more than `max_inputs` buffers, and one of reshapes alone (emitted plain, each is a free view of its operand's buffer; as a fusion it would be a copy kernel);
  `row_dots` moves a GEMM and its epilogue into the row fusion reading it when `row_dot` accepts the body
  (reverting otherwise); `emitted` rebuilds the graph with `Fusion` nodes and `FusionOutput`s for hosted
  values, each step after the values it reads.
- `cuda.rs`, `mps.rs`, `triton.rs` — each device's `Fuses`: the shared driver with its `Backend` and kernel names.
  MPS's also runs horizontal fusion (`horizontal_candidate`, `HORIZONTAL_BUFFERS`) and exports `fuse` for tests.
- `tpu.rs` — TPU's `Fuses`, fusion by planned groups (TPU's own, not the shared driver): plans groups back to front,
  splitting off layout ops until every group's indexing can be planned, and turns the groups into
  `elementwise` or `reduction` fusions (a reducing member leads). A slice alone in its group stays out, for
  `place_views` to view or `alone` to wrap; so does a reshape alone that keeps every element in its tile (a leading
  unit axis added or dropped), which the plan runs as a view of its operand's buffer.

The `Fusion` pass (in `mod.rs`) runs each device's fusion, `Fuses::fuse`: MPS, Triton and CUDA with `fuse` (their
lists' condition) and TPU always. Which nodes each backend's kernels fuse (`fusible`) and how they are named live in
`codegen/fusion`. The CPU doesn't run it.

## Backend

A backend fills `Backend`: `fusible` (which nodes may join a fusion), `max_inputs`, `max_aux`,
`multi_output_fusion`, `reduction_epilogues`, `contraction_epilogues`, `dot_epilogue` (which dots run an
epilogue), `expansion` (whether a GEMM epilogue may write into a concatenation's slot), `row_dot`
(whether a GEMM plus row body runs as one kernel), `swap_epilogues` (whether a GEMM epilogue may take a
`SwapPairs` of distance 1, swapping the accumulator's adjacent lanes in registers: Triton yes, MPS not, its
epilogue functor seeing one element; other distances run after the GEMM), `combine_epilogues` (whether a split-K combine runs its GEMM's epilogue: Triton yes,
MPS and CUDA not) and `update_epilogues` (whether an epilogue writes straight into an update's slot: Triton
yes, MPS and CUDA not).

- MPS: with `fuse`. Elementwise, layout, gather and reduction ops under 2^32 elements, never float64; 29
  input buffers; flags from config; `row_dot` limited to rows of at most `ROW_TILE` unless
  `linear_reordering` is on. Every fusion is requested as a `Kernel::Fusion`. It also runs horizontal
  fusion inside this pass when `horizontal_fusion` is on (MPS has no separate horizontal slot), and repeats
  from scratch until every by-value scalar is read only by fusions.
- Triton: with `fuse`. Ops with Triton dtypes (adds `DynamicSlice`), any number of inputs, reduction and
  contraction epilogues always on, `row_dot` = `dot_row_fits`. Fusions are named by kind (`named`):
  `dot_row`, `row`, `dot_general`, `combine` (a split-K combine, `kinds::combine_root`), `reduction`,
  `elementwise` (`codegen::fusion::triton`'s `fusible` and `named`).
- CUDA: with `fuse`. Elementwise and layout ops (and gathers) with CuTe dtypes only; no reduction epilogues and no rows;
  `dot_epilogue` from `codegen::dot_general::cuda::epilogue`; fusions named `dot_general` or `elementwise`.
- TPU: always, by planned groups (`tpu.rs`). TPU's dot fusions live in `passes/dot`.

In-place concatenations are not fusible (`Analysis::fusible`), so `place_concats` must run first.
