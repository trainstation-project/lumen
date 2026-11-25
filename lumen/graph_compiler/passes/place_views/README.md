# passes/place_views

The `place_views` pass, after fusion: a slice read only by GEMMs that read it in place becomes a strided view
instead of a copy.

## Files

- `mod.rs` - the rule (`fusion_reads_strided`, which MPS codegen calls too), the `StridedOperands` trait (the
  operand layouts a device's GEMM reads in place) and the `PlaceViews` pass.
- `mps.rs`, `triton.rs`, `tpu.rs` - each device's `StridedOperands`: MPS's `reads_strided`, Triton's any strides,
  TPU's slices of whole tiles (`dot_general::tpu::slice_view`).
