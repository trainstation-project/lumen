# passes/diamonds

The `normalization_diamonds` passes: they find normalization rows (row reductions broadcast back over their rows,
softmax and norms) so each becomes one row kernel.

## Files

- `mod.rs` - the matching (`diamonds`, `Row`, `Partials`, helpers codegen shares: `depends_on`, `softmax_pair`,
  `tile_reductions`), the `NormalizationRows` trait and `Diamonds` pass (rows kept for the fusion pass, MPS and
  Triton, with `fuse`), and the `FusedRows` trait and `FusedDiamonds` pass (each row replaced by a `row` fusion,
  TPU's).
- `mps.rs`, `triton.rs` - their `NormalizationRows`: what their fusion fuses, and their row kernels' `Partials`.
- `tpu.rs` - its `FusedRows`: the nodes its row kernel takes and the bodies that fit it.
