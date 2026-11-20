# passes/attention

Flash attention. These passes find attention, forward and backward, written out as primitives in a traced
graph, and replace each match with a fusion kernel (`flash_attention`, `flash_attention_backward`) that
codegen compiles into one attention kernel. They run early in the device pipeline: after the dot merges,
before `linear_cross_entropy` and `simplify` (`matched: true`).

## Files

- `mod.rs` - what both directions share:
  - `Matcher`: producers, live readers and graph outputs of every value; it walks scores, causal masks,
    dropout masks and strided operand views.
  - `Access`: how a kernel reads or writes a `[batch.., row, col]` operand as a strided view of a base
    value (offset, per-batch `(stride, divisor)`, row and column strides). `batch_terms` and `grid_batch`
    turn it into offsets on the kernel grid (outer batch dims merged, heads last; a divisor expresses
    heads sharing a KV head).
  - `Score` (casts and constant scales on the scores), `Dropout` (from `random_bits` against a threshold),
    `Strides` (body inputs read through views and outputs written into views), `body_of` (extract the
    subgraph between operands and outputs as a fusion body), and the names `FORWARD` and `BACKWARD`.
- `forward.rs` - matching `softmax(scores(Q K^T)) V` and the `FlashAttention` pass, with its `Attentions`
  trait: each backend's answers (MPS, Triton, CUDA and TPU, in their codegen; `WITH_FUSE` off on TPU).
- `backward.rs` - matching the gradients and the `FlashAttentionBackward` pass, with its `AttentionBackwards`
  trait: MPS, Triton and TPU answer it in their codegen. CUDA has no backward.
- `layouts.rs` - the `AttentionLayouts` pass (`attention_layouts`), on devices whose codegen implements
  `codegen::TiledAttentions` (TPU), right after the attention passes. It merges the last two axes of outputs
  written with rows narrower than their last axis, folds reshape, sum and cast chains that read attention
  outputs into the fusion bodies, moves the backward's `delta` dot into the backward fusion (`delta_inside`), and
  checks that the backend's attention kernels take every attention fusion.

Gates: both directions run with `flash_attention`, and `fuse` too on MPS, Triton and CUDA (their
`WITH_FUSE`). CPU gets the identity pass for all of them.

## Invariants

- Fusion bodies keep the matched subgraph. Codegen re-matches it from the body with `of_body`,
  `of_body_strided`, `backward_of_body` and its variants, so the matcher is the single source of what a
  kernel computes.
- Operands are read in place as strided views when the matcher can express them as an `Access`; offsets
  must fit in 32 bits (`fits32`).
- Liveness is computed from the graph outputs. A graph with no outputs (still being traced) treats every
  node as live, which `forward::traced` relies on.
