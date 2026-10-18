# `lumen/ops/dot_general`

Kernels for the `dot_general` primitive (XLA's batched, generalized
matmul) and for `linear_cross_entropy` (the logits `h · wᵀ` fused with a
cross-entropy, never materialized). There is no `Op` here: `dot_general` is
a graph primitive, run by a compiled plan. On MPS, `ops::mps::encode` sends
its steps to `mps::encode` / `mps::encode_linear_cross_entropy`; on a TPU,
`ops::tpu::encode` sends `dot_general` to `tpu::encode`; elsewhere the plan
evaluates it with `ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` and `#[cfg(lumen_tpu_linked)] mod tpu;` hooks. |
| `tpu.rs` | `dot_general` on TPU: its form (`form`: batch, then `[M, K] · [K, N]` or `[N, K]`), block choice (`block`), and the Mosaic kernel (`kernel`); see *TPU* below. |

## TPU

`tpu.rs` writes one Mosaic kernel per dot ([`../tpu/`](../tpu/README.md)
runs it). The lhs must be batch, free then contracting dimensions; the rhs
batch, then contracting and free dimensions (`[K, N]`) or free and
contracting (`[N, K]`, `transposed`); anything else errors. Operands are
float32 or bfloat16 (one dtype), accumulating in float32; the output
float32 or bfloat16.

* **Grid** `(batch, M/bm, N/bn, K/bk)`: each step DMAs its lhs and rhs
  blocks to VMEM, multiplies them on the matrix unit into float32
  accumulators (VMEM, zeroed at the first K block), and the last K block
  rounds and DMAs the output block back. DMAs wait as they go: no overlap
  with compute yet.
* **Blocks**: `bm` is M if at most 512, else a multiple of 8 dividing it;
  `bn` and `bk` the largest multiples of their unit (128 words of a row) up
  to 512 (K: 512 · elements a word) dividing N and K. Shapes that cannot
  block error, naming the dimension.
* **Words.** The memory holds tensors row-major, as the host does; the
  kernel reads 32-bit words. A bfloat16 word holds two neighbours of a row,
  so a bfloat16 dot splits operands by parity: `out[:, q::2] = Σ_p
  lhs[:, p::2] · rhs[p::2, q::2]` (a rhs `[K, N]` pairs its rows by k
  parity: two rows a row of `[K/2, N]` words; a `[N, K]` one by n parity),
  each half widened to float32 by a shift and narrowed to bfloat16 exactly.
  A bfloat16 output is written as words: even columns' rounded bits in the
  low halves, odd columns' in the high.
* **Supported**: float32 → float32; bfloat16 → bfloat16; bfloat16 → float32
  with a `[N, K]` rhs. The others (float32 → bfloat16, bfloat16 → float32
  with a `[K, N]` rhs) would interleave columns across lanes: not yet.
* float32 operands multiply in full float32 (`contract_precision<fp32>`),
  not the matrix unit's bfloat16 passes.

## Subfolders

* [`mps/`](mps/README.md) — the Metal matmul kernels (integer, 64-bit
  integer, simdgroup-matrix float, split-K atomic, linear-cross-entropy) and
  the host code that canonicalizes a `dot_general` to `[b, m, k] × [b, k, n]`
  and picks a kernel and tile.

## Related

* `lumen/graph_compiler/mps/canonicalize_dots.rs`, `split_k.rs`,
  `place_views.rs` — compiler passes that use this folder's matmul-form
  helpers.
* `lumen/codegen/mps/mod.rs` — fusion kernels that call the `matmul_sg_impl` template.
