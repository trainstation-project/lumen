# dot

Passes that rewrite dots (`Primitive::DotGeneral`): folding reshapes into them, strength-reducing vector
dots, splitting long contractions, putting operands in a layout the matmul kernels read, and fusing a
matmul with what surrounds it. Each file is its own pass, which device lists add where they run it; a pass with
per-device answers defines a trait each device implements in its file here (`mps.rs`, `triton.rs`, `tpu.rs`).

## Files

| File | Pass | Trait | Devices |
|---|---|---|---|
| `fold_reshapes.rs` | `FoldDotReshapes`: `dot(reshape(x), w)` becomes `reshape(dot(x, w))` | none | TPU |
| `strength.rs` | `DotStrength`: dots between vectors become `reduce_sum(x * y)` | none | MPS, Triton, TPU |
| `split_k.rs` | `SplitK` (with `split_k`): long-K matmuls with few output tiles become a batched dot over K chunks plus a sum | `SplitsK` (tiles, target, kept whole, preferred chunks) | MPS, Triton |
| `canonicalize.rs` | `CanonicalizeDots`: transposes operands a matmul kernel can't read as laid out; also exports `matmul_order` (MPS's GEMM codegen) | `DotLayouts` (`readable`) | MPS, TPU |
| `rows.rs` | `DotRows` (with `normalization_diamonds`): a matmul fused into the `row` fusion reading its whole rows; the shared `dot_rows` | `FullRows` (`full`) | TPU |
| `wide_rows.rs` | `WideRows` (with `fuse`, `normalization_diamonds` and `linear_reordering`): rows too wide for one GEMM tile get the GEMM writing their reductions' partials | `PartialRows` (`partial_tile`, `gemv`) | Triton |
| `fuse.rs` | `FuseDots`: dot fusion with slices, a stacked lhs, epilogues and heads (the reshapes and transposes splitting a value into attention heads, and KV cache updates, written by the GEMM) | `FusesDots` (`LANES`, `slot_writes`) | TPU |

- `mps.rs`, `triton.rs`, `tpu.rs` - each device's answers: MPS's `DotLayouts` and `SplitsK`, Triton's `SplitsK` and
  `PartialRows`, TPU's `DotLayouts`, `FusesDots` and `FullRows`.

## Order in the pass lists

These slots run in this order, interleaved with other passes:

1. `fold_dot_reshapes`, right after `linear_cross_entropy` and before `Simplify`.
2. `dot_strength`, after `Simplify`.
3. `split_k`, after `linear_reordering`.
4. `canonicalize_dots`, after `split_k` and before the diamonds passes.
5. `dot_rows` (TPU), right after `fused_diamonds`, whose `row` fusions it reads.
6. `fuse_dots` (TPU), after `diamonds` and before `place_concats` and `fusion`.
7. `wide_rows` (Triton), right after `fusion`; it reads the `row` and GEMM fusions fusion made.

CPU and CUDA run none of these.
