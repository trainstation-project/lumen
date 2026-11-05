"""The CUDA (CuTe DSL) backend's GEMMs: CUTLASS's Ampere tensor-core GEMM (`gemm_sm80`), and `launch`, which compiles
it for a matmul's dtypes and layouts and runs it on lumen tensors."""
