# `lumen/kernels/attention/mps`

Flash-attention as Metal **templates** (no `kernel` functions). They are
pasted into the prelude of generated fusion kernels
(`graph_compiler/mps/mod.rs`, `PRELUDE`, after `kernels/mps/kernels.metal`,
reduce and dot_general), and `codegen/mps/mod.rs` writes a small kernel per
matched attention that instantiates one of them with the shapes, dtypes and
functors baked in. Not part of the primitive library built by `build.rs`.

## Files

| File | Contents |
| --- | --- |
| `forward.metal` | `NoDropout`; `ATTN_ROWS`/`ATTN_GROUPS`/`ATTN_BQ`; `flash_attention` (tiled, simdgroup matrices) and `attention_decode` (few queries). |
| `backward.metal` | `flash_attention_dkdv` (dK and dV per key block, optionally dQ by atomics) and `flash_attention_dq` (dQ per query block). |

## Template parameters

The generated kernel supplies:

* `T` — the q/k/v dtype; `O` — the attention output's dtype; `Out` and `Epi`
  — a fused epilogue (`Same` if none) writing the fusion's output.
* `D`, `DV` — head dims of q/k and v; `BK` (forward) / `BQ` (backward) — the
  block staged in threadgroup memory.
* `CAUSAL` with runtime `offset`: key `j` is seen by query `i` iff
  `j <= i + offset`.
* `Ix` — an indexer struct: per-batch base offsets (`ix.q(b)`, …) and
  static row/column strides (`Ix::Q_ROW`, `Ix::Q_COL`, …), so q/k/v are read
  as strided views (transposes, slices, grouped-query broadcasts).
* `Score` — the scale and casts applied to a raw score; `Drop` — dropout
  (`NoDropout` or a Philox-based functor from codegen).
* `RAW` writes the un-epilogued output too; `LSE` writes the per-row
  log-sum-exp (float), which the backward reads.

## Kernels

* **`flash_attention`**: a threadgroup of 256 threads (8 SIMD groups) owns
  `ATTN_BQ = 64` query rows, 8 per SIMD group. It loops over key blocks of
  `BK`, staging K and V in threadgroup memory, computing scores on 8×8
  simdgroup matrices accumulated in float, with an online softmax (running
  max `m` and sum `l`). Probabilities are rounded to `T` before multiplying
  V, and normalized by `l` at the end. Causal blocks stop at the last
  visible key.
* **`attention_decode`**: one threadgroup per query row (chosen when
  `sq <= 8`, `codegen::decodes`); each SIMD group takes every 8th key, lanes
  split the head dim, and the 8 partial (max, sum, output) triples are
  merged in threadgroup memory.
* **`flash_attention_dkdv`**: a threadgroup per block of 64 keys, looping
  over query blocks of `BQ`; recomputes P from `lse`, uses `delta` for dS,
  accumulates dK and dV in float. With `DQ`, it also adds dS·K into a float
  dQ with `atomic_fetch_add` (relaxed, so in no fixed order).
* **`flash_attention_dq`**: a threadgroup per 64 queries, looping over key
  blocks of `BK`, accumulating dQ in float; used when dQ is not done by
  atomics.

## Choosing the variant (in `codegen/mps/mod.rs`)

* Forward: `decodes(a)` (`sq <= 8`) picks `attention_decode`; else
  `flash_attention` with `key_block`, the largest of 32/16/8 keys whose K
  and V blocks fit in 28 KB of threadgroup memory.
* Backward: `backward_block_with_dq` picks 32 or 16 queries if Q, dO, K and
  dSᵀ fit in 28 KB (one kernel, dQ by atomics); otherwise two kernels,
  `flash_attention_dkdv` without dQ and `flash_attention_dq`, with
  `backward_block` (32/16/8).

## Invariants & gotchas

* `D`, `DV`, `BK`, `BQ` must be multiples of 8 (8×8 fragments, `frag_row`/
  `frag_col` from `kernels/mps/kernels.metal`).
* Threads index staging loops with `t < 256`: the launch must be
  16 × 16 threadgroups (`Grid::Groups`).
* Scores, softmax and both dots accumulate in float; the matcher only
  accepts traces that compute so.

## Related

* [`../../mps/README.md`](../../mps/README.md) — shared prelude and launch.
* `lumen/graph_compiler/common/attention.rs` — what gets matched and its exact rounding.
* [`../../dot_general/mps/README.md`](../../dot_general/mps/README.md) — the same simdgroup-matrix style for matmuls.
