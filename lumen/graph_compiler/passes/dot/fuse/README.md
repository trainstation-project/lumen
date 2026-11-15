# fuse

TPU's dot fusion: `tpu::FuseDots` (pass name `fuse_dots`) replaces a matmul and the work around it with
one `dot_general` fusion that the TPU matmul kernel runs. It runs after `diamonds` and before
`place_concats` and the general `fusion` pass, which then handles what is left.

## Files

- `mod.rs` — declares the `tpu` module only.
- `tpu.rs` — `FuseDots` and its planner. TPU runs it always.

Every other device gets the identity pass (MPS, Triton and CUDA put GEMM epilogues in the shared fusion
pass instead, see `passes/fusion`).

## What one dot fusion absorbs

For each `DotGeneral`, the planner (`fused`) collects:

- Splitting slices: when the dot is no graph output and is read only by `Slice`s that tile one of its
  last two axes end to end, each keeping the other axes whole, with bounds on `LANES` boundaries. The
  slices are rebuilt inside the body.
- A stacked lhs: when the lhs is a `Concatenate` along one of its last two axes of lane-aligned parts,
  read only by dots as their lhs, not a graph output, and not an in-place concatenation (checked with
  `place_concats::in_place_concats` and `fusion::tpu::slot_writes`). The fusion reads the concat's parts.
- Elementwise epilogues, only with `contraction_epilogues`: later elementwise readers of the dot (or of a
  slice, or of an earlier epilogue), claimed greedily in node order. Each must have its part's shape; its
  other operands must be constants (rebuilt in the body), scalar broadcasts, or scalar or same-shape
  values computed before the dot. Epilogues may combine several slices only when those are at most 4
  equal-width slices along the last axis; otherwise a non-scalar extra operand serves one slice only.

A dot with nothing to absorb, or whose members are not read outside the fusion, stays as it is.

## Output

Each planned dot becomes a `dot_general` fusion whose outputs are the member values read outside it, in
variable order; the first is the fusion's value and the rest are read through `FusionOutput` nodes. Its
inputs are the dot's operands (or the stacked concat's parts) followed by the epilogues' extra operands.
Its label is the dot's label, prefixed `Nx` when an unlabelled dot stands for N slices or N stacked parts,
followed by the absorbed nodes' labels.
