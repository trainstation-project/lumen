# linear_cross_entropy

`LinearCrossEntropy` (pass name `linear_cross_entropy`) replaces an unfused cross entropy over logits
`h @ w` with one `Primitive::LinearCrossEntropy` op that never materializes the `[b, v]` logits. It runs
after the attention passes and before `fold_dot_reshapes`; `chunked_cross_entropy`, earlier in the list,
reuses its matcher.

## Files

- `mod.rs` — the matcher and the device-agnostic pass.
- `mps.rs` — MPS runs it with `fuse` and `fused_linear_cross_entropy` (built only when MPS is linked).
- `triton.rs` — Triton runs it with `fuse` and `fused_linear_cross_entropy`.

CPU, CUDA and TPU get the identity pass.

## Pattern

For an unbatched dot computing float32 logits (float32 accumulation and output) from 2-D hidden states
`h` (`[b, d]`, contracted on dim 1) and a 2-D weight `w` of the same float dtype (`[v, d]`, or `[d, v]`
when the dot contracts the weight's dim 0, `transposed`):

- a max-shifted logsumexp along the classes: `ReduceMax` over axis 1, `Sub` of its broadcast, `Exp`,
  float32 `ReduceSum` over axis 1, `Log`, `Add` of the row max (`lse`). The logits may be read only by the
  row max, the shift and a flattening reshape; each intermediate only by the next step.
- for a loss, the flattened logits gathered (`Gather { axis: 0 }`) at `iota(b) * v + target`, with
  `target` an int32/int64 `[b]` tensor defined before the fused op is emitted, and `lse - picked` as the
  per-row loss.

## Rewrite

- A full match emits, at the earliest of the lse and the gather, `LinearCrossEntropy(h, w, target)`
  (with `w` transposed to `[v, d]` first when needed). Its `[2, b]` result holds the loss row (0) and the
  lse row (1); each is sliced and reshaped to `[b]` and replaces the old loss and lse values.
- A logsumexp whose logits nothing else reads (a gradient's, with the loss unused) is replaced by the
  same op with all-zero int32 targets, using only its lse row.

Every other node is copied with its label; replaced nodes pass their origins to the new ones.

## Shared with `chunked_cross_entropy`

`Found`, `matched(graph, producer, readers, dot)` and `uses(graph)` are `pub(super)` so
`passes/chunked_cross_entropy` can match the same pattern.
