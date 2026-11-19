# dot

Passes that rewrite dots (`Primitive::DotGeneral`): folding reshapes into them, strength-reducing vector
dots, splitting long contractions, putting operands in a layout the matmul kernels read, and fusing a
matmul with what surrounds it. This folder only groups them; each subfolder is its own pass with its own
slot in the one pass list (`graph_compiler::pipeline::pipeline!`).

## Subfolders

- `fold_reshapes/` — `FoldDotReshapes`: `dot(reshape(x), w)` becomes `reshape(dot(x, w))`. TPU only.
- `strength/` — `DotStrength`: dots between vectors become `reduce_sum(x * y)`. MPS, Triton, TPU.
- `split_k/` — `SplitK`: long-K matmuls with few output tiles become a batched dot over K chunks plus a
  sum. MPS only, behind `split_k` (Triton's GEMM step splits K itself).
- `canonicalize/` — `CanonicalizeDots`: transposes operands a backend's matmul kernel can't read as laid
  out. MPS and TPU. Also exports `matmul_order`, used by MPS's GEMM codegen.
- `rows/` — fuses a matmul into the normalization `row` fusion that reads its output: `tpu::DotRows`
  (TPU) and `triton::WideRows` (Triton).
- `fuse/` — `tpu::FuseDots`: TPU's own dot fusion with slices, a stacked lhs and epilogues.

## Order in the pipeline

These slots run in this order, interleaved with other passes:

1. `fold_dot_reshapes`, right after `linear_cross_entropy` and before `Simplify`.
2. `dot_strength`, after `Simplify`.
3. `split_k`, after `linear_reordering`.
4. `canonicalize_dots`, after `split_k` and before the diamonds passes.
5. `dot_rows` (TPU), right after `fused_diamonds`, whose `row` fusions it reads.
6. `fuse_dots` (TPU), after `diamonds` and before `place_concats` and `fusion`.
7. `wide_rows` (Triton), right after `fusion`; it reads the `row` and GEMM fusions fusion made.

CPU and CUDA run none of these (identity passes).
