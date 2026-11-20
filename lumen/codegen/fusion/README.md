# fusion

Each backend's fusion: which nodes its kernels fuse, its view of the shared fusion (`passes::fusion::Backend`) or
a fusion of its own, how its fused bodies are named, and the helpers other passes and codegen ask. Each implements
`codegen::Fuses`, which the graph compiler's `fusion` pass (`passes/fusion`) runs.

## Files

- `mod.rs` — declares the backends.
- `mps.rs` — MPS (built only when MPS is linked): `fusible` (elementwise, layout, gather and reduction ops under
  2^32 elements, never float64), its `Backend` (29 input buffers, `row_dot` limited to `ROW_TILE` without
  `linear_reordering`), and its `Fuses`: the shared fusion, every fusion requested as a `Kernel::Fusion`, then
  horizontal fusion within the pass with `horizontal_fusion`, repeated until every by-value scalar is read only
  by fusions. With `fuse`. Also which inputs MPS kernels read through views: `writes_views`, `loads_strided`,
  `loads_views` and `fusion_reads_strided` (a GEMM fusion input its matmul strides through).
- `triton.rs` — Triton: `fusible` (ops with Triton dtypes), its `Backend` (any number of inputs, every epilogue
  kind on, `row_dot` = `dot_row_fits` but not GEMVs), `named` (`dot_row`, `row`, `dot_general`, `combine`,
  `reduction`, `elementwise`), `horizontal_candidate`, `HORIZONTAL_BUFFERS` and `slot_writes`. With `fuse`.
- `cuda.rs` — CUDA (CuTe DSL): `fusible` (elementwise and layout ops and gathers with CuTe dtypes), its `Backend`
  (no reduction epilogues, no rows, `dot_epilogue` from `dot_general::cuda::epilogue`), fusions named
  `dot_general` or `elementwise`. With `fuse`.
- `tpu.rs` — TPU: its `Fuses` runs `passes::fusion::groups` (planned groups), always; `horizontal_candidate`,
  `HORIZONTAL_BUFFERS` and `slot_writes`.
