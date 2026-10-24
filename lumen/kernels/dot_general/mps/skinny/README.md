# `lumen/kernels/dot_general/mps/skinny`

The `Skinny` algorithm: a matmul of a few rows against a wide matrix (decode-style), a tuning candidate for float GEMM fusions of 32 rows or fewer with a plain (elementwise) epilogue, a dot alone included.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `skinny` (32 rows or fewer, B contiguous along N, and at least 2^24 multiply-adds, below which the fixed default runs). |
| `config.rs` | `skinny_configs` (1 to 8 rows, 128 or 256 threads, 32- to 256-column blocks, 4 or 8 columns per thread). |
| `matmul_skinny.metal` | `matmul_skinny`, for a few rows (PyTorch's `gemv_t2d`, for up to 8 rows): a threadgroup takes a block of columns, its threads 4 or 8 consecutive columns each (16-byte rows of B) and the rest of them split K, accumulating every row in float32; the parts meet in threadgroup memory. Its optional `Epi` is applied to each element as it is stored (the fused epilogue). |
