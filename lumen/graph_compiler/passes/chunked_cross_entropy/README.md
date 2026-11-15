# passes/chunked_cross_entropy

The `chunked_cross_entropy` pass. Sometimes a graph computes a cross entropy over the full logits `h @ w`
and also computes the same logits chunk by chunk from row slices of `h`. This pass rewrites the cross
entropy to run per chunk on the chunked logits, so the full logits are no longer needed. It runs early in
the device pipeline, after `take_along_axis` and before the dot merges and `linear_cross_entropy`.

## Pattern

- The cross entropy is the one `linear_cross_entropy::matched` finds on dot node `h @ w`: a max-shifted
  `logsumexp` over the classes, and the target logits gathered from the flattened logits.
- The chunks are other dots with the same primitive and the same weight whose lhs is a `Slice` of `h`
  covering every column, `[start, end)` in rows.
- The chunks must tile rows `0..b` without gaps or overlap.

## Rewrite

- The softmax region, plus the gather-index computation above the target, is replayed per chunk on that
  chunk's rows. `of_rows` restates each primitive for a chunk of `c` of the `b` rows.
- The per-row lse and loss of all chunks are joined by concatenation along dim 0, at most 16 operands
  per concatenate (`joined`).
- A slice of the lse or loss matching one chunk's rows reads that chunk's value directly.
- Readers of the joined lse or loss go after the last chunk.
- The match is skipped when the chunks don't tile the rows, or when the rewrite would change the order
  of reads. The pass repeats until no match is left.

## Files

- `mod.rs` - `chunked_cross_entropy`, `chunked`, `joined`, `of_rows` and the `ChunkedCrossEntropy` pass,
  generic over any `Module`.
- `mps.rs` - MPS runs it when `fuse` and `fused_linear_cross_entropy` are on.
- `triton.rs` - Triton runs it under the same two flags.
- CUDA, TPU and CPU: identity.
