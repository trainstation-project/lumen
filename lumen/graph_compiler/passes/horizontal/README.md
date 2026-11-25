# passes/horizontal

The `horizontal_fusion` pass, last in the lists that run it: merges independent small kernels of the same kind into
one launch (`fuse(graph, scalars, target)`, grouping candidates by key while the graph stays acyclic and the merged
kernel fits the device's buffer limit).

## Files

- `mod.rs` - the grouping and merging (`fuse`, `Grouping`, `Target`), the `HorizontallyFuses` trait and the
  `HorizontalFusion` pass (with `horizontal_fusion`).
- `triton.rs` - Triton's candidates (non-trivial `elementwise` fusions without reductions or matmuls), tiling
  grouping, 64 buffers; with `fuse`.
- `tpu.rs` - TPU's candidates (non-scalar `elementwise` fusions of TPU-fusible nodes not written into a concat),
  grouping by shape, 32 buffers; always.

MPS merges kernels inside its own fusion pass instead (`fusion/mps.rs`).
