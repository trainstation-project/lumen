# `lumen/kernels/dot_general/mps/tiled`

The float matmuls on simdgroup matrices: the precompiled `matmul_`, `matmul_mid_` and `matmul_small_` kernels the untuned `dot_general` runs, the split-K atomic kernel, and the tuned `Tiled` candidates. `codegen/mps` instantiates the same `matmul_sg_impl` for GEMM fusions with an epilogue.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `source`, the generated kernel of a tuned `Tiled` config: `matmul_sg_impl` with its tile, threads, SIMD-group columns and prefetch. |
| `config.rs` | `configs` (the valid candidates: whole 8×8 fragments per SIMD group, 4 to 16 of them and roughly square, even loads, within 32 KB of threadgroup memory); `DEFAULT_CONFIGS` (the three precompiled variants). |
| `matmul_sg.metal` | `matmul_sg_impl` (floats on simdgroup matrices, with `ATOMIC`/`PAIRED`/`EXPANDED`/`TILE` modes), its threadgroup sizes (`SG_LT`, `SG_LT_OF`) and the float kernels (`matmul_`, `_mid_`, `_small_`, widened). As in MLX's steel GEMM: when both operands are contiguous along their loads and aligned, each thread copies 16 bytes at a time (`Bytes`) from advancing pointers, unpredicated on whole tiles; the multiply walks each SIMD group's fragments in serpentine order between `simdgroup_barrier`s. Otherwise it loads element by element. |
| `matmul_atomic.metal` | `matmul_atomic_`, the split-K kernel accumulating float partials with atomics. |
