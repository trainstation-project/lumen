# place_views

`PlaceViews` (pass name `place_views`) turns slices read only by GEMMs into strided views of their operand
(`Slice { view: true }`) instead of copies, when the GEMM kernel can read the operand at the parent's
strides. It runs after `fusion` and `wide_rows` and before `alone`, so the lone layout ops `alone` turns
into elementwise fusions no longer include these slices.

## Files

- `mod.rs` — `ReadsStrided`, `place_views`, `fusion_reads_strided` and the `PlaceViews` pass.
- `mps.rs` — MPS runs it always, with the MPS GEMM's `reads_strided` rule. Built only when MPS is linked.
- `triton.rs` — Triton runs it always; Triton's GEMM reads its operands at any strides.

CPU, CUDA and TPU get the identity pass.

## Rule

A `Slice` of one value is marked a view when:

- its output is not a graph output and its input is not itself a slice already marked a view;
- it has at least one reader, and every reader is either a `DotGeneral` whose `reads(dot, operands, k,
  strides)` accepts each operand slot reading the slice at the sliced input's contiguous strides, or a GEMM
  fusion where `fusion_reads_strided` holds.

`fusion_reads_strided(body, k, strides, reads)` holds when fusion input `k` is read only by the body's GEMM
dot (`codegen::kinds::gemm_dot`), as exactly one of its operands, in a layout `reads` accepts.

`ReadsStrided` is `fn(&Primitive, [&TensorType; 2], operand, strides) -> bool`, the backend's GEMM rule.
`fusion_reads_strided` is also called by MPS codegen (`codegen::mps::fusion_reads_strided`, used when
planning MPS steps).
