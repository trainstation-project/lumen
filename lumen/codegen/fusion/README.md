# fusion

Which nodes each backend's kernels fuse, how its fused bodies are named and which inputs they read through views:
what codegen and the graph compiler's passes both ask. Each backend's fusion itself (`Fuses`, its `Backend`) is in
`graph_compiler/passes/fusion/<device>.rs`; TPU's fusible nodes are `codegen::devices::tpu::fusible`.

## Files

- `mod.rs` — declares the backends.
- `mps.rs` — MPS (built only when MPS is linked): `fusible` (elementwise, layout, gather and reduction ops under
  2^32 elements, never float64), and which inputs MPS kernels read through views: `writes_views`, `loads_strided`,
  `loads_views` and `fusion_reads_strided` (a GEMM fusion input its matmul strides through).
- `triton.rs` — Triton: `fusible` (ops with Triton dtypes) and `named` (`dot_row`, `row`, `dot_general`, `combine`,
  `reduction`, `elementwise`).
- `cuda.rs` — CUDA (CuTe DSL): `fusible` (elementwise and layout ops and gathers with CuTe dtypes) and `ELEMENTWISE`,
  the name of its elementwise fusions.
