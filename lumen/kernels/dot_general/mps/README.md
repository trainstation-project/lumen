# `lumen/kernels/dot_general/mps`

Metal matmul kernels for `dot_general` and `linear_cross_entropy`, and the
host side that maps a `dot_general` onto them. Called from
`kernels::mps::encode` for plan steps; the same kernels and helpers are reused
by `graph_compiler/mps` (split-K dots, view placement, dot
canonicalization) and by generated fusion kernels in `codegen/mps`, which
instantiate the `matmul_sg_impl` template with an epilogue.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Matmul form (`matmul_order`, `collapsed`, `reads_strided`), tile choice (`float_tiles`, `float_groups`; a tuned `Tile` from `tile_for`, the `GemmAutotuner`'s pick or the one being timed, `trying`), `plan_matmul` → `MatmulLaunch`, `encode`, and `linear_cross_entropy`'s `encode_linear_cross_entropy` / `linear_cross_entropy_scratch`. |
| `config.rs` | GEMM tuning: `GemmConfig` (tile M, N and K step, threads, SIMD-group columns), `configs` (the valid candidates: whole 8×8 fragments per SIMD group, 4 to 16 of them and roughly square, even loads, within 32 KB of threadgroup memory), `DEFAULT_CONFIGS` (the three precompiled variants), and the tuned pick's lookup (`tile_key`, `name_config`, `tuned`; `trying` while one is timed). |
| `kernels.metal` | `matmul_impl` (8–32-bit ints), `matmul_wide_impl` (64-bit ints), `matmul_sg_impl` (floats on simdgroup matrices, with `ATOMIC`/`PAIRED`/`EXPANDED`/`TILE` modes), `LceRows`, and the kernel instantiations incl. `lce_combine`. |

## Matmul form

Every `dot_general` runs as `out[b, m, n] = Σ_k lhs[b, m, k] · rhs[b, k, n]`.
`matmul_order` orders the lhs dims as *batch, free, contracting* and the rhs
dims as *batch, contracting, free*. `collapsed` then checks that each of the
three groups collapses to one strided dimension (ignoring size-1 dims) and
returns its stride; if any group does not, `plan_matmul` errors ("an
operand is not in matmul form"). The compiler's MPS passes make this hold
(`canonicalize_dots.rs` inserts transposes; `place_views.rs` lets an operand
stay a strided view only if `reads_strided` says the kernel can read it).

Kernels get one `constant ulong *p` of 9 values:
`[M, N, K, lhs_b, lhs_m, lhs_k, rhs_b, rhs_k, rhs_n]` (sizes, then element
strides), and three buffers `lhs, rhs, out` (`out` is contiguous).

## Kernel variants and tiles

All launch as `Grid::Groups([ceil(N/tn), ceil(M/tm), batch])` with
16 × 16 = 256-thread threadgroups.

| Output dtype | Kernel | Tile (M × N) | Notes |
| --- | --- | --- | --- |
| 8–32-bit ints | `matmul_<dtype>` | 64 × 64, K step 16 | each thread 4×4 outputs; double-buffered loads |
| `u64`, `i64` | `matmul_<dtype>` | 32 × 32 (`WIDE_TILE`) | fewer registers |
| float, large | `matmul_<name>` | 128 × 64, K step 16 (`SG_BK`) | 8 SIMD groups in 4×2, each 32×32 of 8×8 accumulators |
| float, `M % 128` in 33..=64 | `matmul_mid_<name>` | 64 × 64 | |
| float, few tiles | `matmul_small_<name>` | 32 × 32, K step 32 (`SMALL_BK`) | fewer than 32 large tiles, or `M % 128` in 1..=32 |

`<name>` is the operand dtype when lhs, output and `accum_dtype` agree
(`f16`, `bf16`, `f32`: accumulate in the operand dtype), else
`<lhs>_<accum>_<out>`; only `f16_f32_{f16,f32}` and `bf16_f32_{bf16,f32}`
exist (`MATMUL_WIDENED`), i.e. half/bfloat operands accumulating in float.
Integer kernels accumulate in their own dtype (wrapping).

`matmul_atomic_<dtype>` (64 × 64, float accumulate and output) is a split-K
dot's: launched by `graph_compiler/mps/mod.rs` (not `encode`) after zeroing
the output, with `batch` = the number of K chunks, each atomically added.

## `linear_cross_entropy`

Two launches, needing scratch of `2 · B · ceil(V / tn) · 4` bytes:

1. `lce_partials_[mid_|small_]<h dtype>_<i32|i64>` — grid
   `[ceil(V/tn), ceil(B/tm), 1]`; a threadgroup computes a logits tile
   (`TILE` mode: accumulators handed to `LceRows`, not stored), writes each
   row's tile max and Σexp to scratch, and the target class's logit to
   `out[0, row]`. Tile size follows `float_tiles(1, B, V)`.
2. `lce_combine` — a SIMD group per row (8 rows a threadgroup) merges the
   partials: `out[1, row]` = log-sum-exp, `out[0, row]` = lse − target logit.

`h` is `[B, D]`, `w` is row-major `[V, D]` read as a transposed rhs. Output
is float `[2, B]`.

## Invariants & gotchas

* Each output sums over k in increasing order (blocks of 8 for floats), so
  results are deterministic — except `matmul_atomic` and anything else that
  adds chunks atomically.
* Operand tiles are zero-padded past the edges; interior tiles skip bounds
  checks.
* Tile selection lives in `float_tiles` (also behind `float_groups`, which
  split-K uses, and the LCE tile); its tile sizes must match the
  `MATMUL_FLOAT` / `MATMUL_WIDENED` / `LCE_PARTIALS` instantiations.
* `out` is written contiguous `[batch, M, N]`; only the operands may be
  strided.
* `kernels.metal` wraps its kernel instantiations in `#ifndef TEMPLATES_ONLY`
  so fusion kernels can include the templates without duplicate kernels.

## Related

* [`../../mps/README.md`](../../mps/README.md) — `launch`, `Grid`, the shared prelude.
* `lumen/graph_compiler/mps/split_k.rs` — when a dot is split along K.
* `lumen/graph_compiler/mps/canonicalize_dots.rs`, `place_views.rs` — getting operands into matmul form.
* `lumen/codegen/mps/mod.rs` — fused GEMM epilogues, paired (SwiGLU) dots.
