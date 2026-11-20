# reduction

Code generation for reduction fusions: one `ReduceSum` or `ReduceMax` (the root) with the elementwise
prologue feeding it and the epilogue after it. Codegen reaches it when `kinds::reduction_root` finds a
single reduction in a fused body; MPS also runs lone reductions here.

## Files

- `mod.rs` — the `ReductionCodegen` trait: `reduction(body, by_value, root)` returns a backend's `Code`
  for the fusion, inputs flagged in `by_value` passed as scalars.
- `triton.rs` — Triton reductions plus the tile and combine helpers that rows and scans reuse.
- `mps/` — Metal reductions (`mod.rs`, `kernel.metal`).
- `tpu/` — Pallas reductions (`mod.rs`, `kernel.py`).

There is no CUDA reduction yet.

## Backends

**Triton (`triton.rs`).** `ReductionCodegen for Triton` generates one kernel per fusion as a `Reduction`
(name, source, `lanes` per output, `splits` per output). When the fusion has fewer than `SPLIT_BELOW`
outputs, the reduced axis is split across programs and a second `NAME_final` kernel combines the
partials. Lane counts depend on whether the reduced axis is innermost (`ROW_LANES`, `SPLIT_ROW_LANES`)
or strided (`COLUMN_LANES`). `codegen/utils/triton::loop_kernel` calls it, and the step runs on the
generic `Loop` kernel (`elementwise/triton`). The shared helpers are `Tile` (a 2-D program tile with
outputs on one axis and reduced lanes on the other), `combine`, `reduce_along` and `lowest`.
`row/triton.rs`, `scan/triton` and `dot_general/triton/row.rs` use them.

`split_k_combine` generates a split-K combine (a `combine` fusion, `kinds::combine_root`): each program
(`combine_blocks` of them) sums one block of the product over its K chunks, in chunk order so the result is
deterministic, then runs the epilogue the unsplit GEMM would (`dot_general::triton::Product::stores`): the
rounding cast, column regions such as a merged projection's members, pair swaps, every output through its
`STORE` helper. Blocks take as many rows as the product has, up to `COMBINE_ROWS`, and `COMBINE_COLUMNS`
columns, so a decode step's single row computes its epilogue once.

**MPS (`mps/`).** `kernel.metal` holds the hand-written templates. `reduce` uses one thread per output
over any strides. `reduce_grouped` uses power-of-two lanes per output and combines them with SIMD
shuffles or threadgroup memory. `reduce_rows` and `reduce_cols` reduce chunks of contiguous rows or
columns into partials. Defining `TEMPLATES_ONLY` keeps the templates and skips the instantiations.
`layout` normalizes the input (it drops size-1 dims and merges adjacent dims) and picks a `Layout`:
`Generic` (beyond 32-bit indexing), `Grouped`, `Rows` or `Cols`. Long reductions split into a pass
that writes partials into scratch and a final pass (`splits`, `scratch_bytes`). Use these functions:

- `encode`: encodes a lone reduction.
- `encode_reduction`: encodes a reduction with a generated fused kernel in place of the built-in ones.
- `ReductionCodegen for Mps`: generates that fused kernel. It reads the operand through a `NAME_input`
  functor and adds a `NAME_final` kernel that applies the epilogue when the reduction splits.

The `Reduction` Kernel struct (config `()`, no candidates) runs lone `reduce_sum`/`reduce_max` steps.
Fused reductions run through `runtime::kernel::mps::Fusion`, which calls `encode_reduction`,
`split_launches` and `has_epilogue`. `accum_dtype` and `kernel_dtype` name the accumulator and the
kernel suffix. `scan/mps` reuses `kernel_dtype`.

**TPU (`tpu/`).** `layout` maps the reduced axes to a `Layout`:

- `Rows`: the last axis.
- `Columns`: every axis but the last.
- `Total`: every axis.

`reducible` says whether the kernel handles a node: non-bool dtypes of 32 bits or less, and a supported
layout. The TPU fusion passes call it to form `reduction` fusions. `ReductionCodegen for Tpu` emits a
`Spec` (kernel name plus JSON with inputs, prologue nodes, `reduce`, epilogue nodes and `result`).
`blocks` sizes the blocks to whole TPU tiles near `BLOCK_BYTES` of VMEM. `kernel(step)` wraps the spec
as a `codegen::utils::tpu::Kernel` of module `lumen.codegen.reduction.tpu.kernel` with `Rows::Last`, and the
`Pallas` kernel launches it. `kernel.py::program(spec)` is the fixed schedule. It reduces operand blocks
into a VMEM accumulator, replays the prologue and epilogue nodes, and runs the epilogue on the last step
of a result block. A 32-bit integer max goes through float32 maxima of 16-bit halves (`_exact_max`).

## Conventions

- `REDUCE_THREADS` in `mps/mod.rs` must match `REDUCE_THREADS` in `kernel.metal`.
- The max identity is `-inf` for floats, the minimum for signed ints and 0 otherwise (`lowest`).
  A float max propagates NaN, and a bool max is `or`.
- A sum accumulates in the `ReduceSum`'s declared accumulator dtype; a max accumulates in the input
  dtype.
