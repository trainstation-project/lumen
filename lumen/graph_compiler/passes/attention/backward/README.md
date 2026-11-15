# passes/attention/backward

The backward flash-attention pass (`FlashAttentionBackward`, named `flash_attention_backward`). It finds the
gradients of an attention that autograd traced as primitives and replaces them with
`flash_attention_backward` fusions. It runs right after the forward `flash_attention` pass.

## Pattern

The match is built around the `exp` node computing the transposed probabilities `P^T = exp(S^T - lse)`:
- `dV = P^T dO`
- `dS = P^T * (V dO^T - delta) * scale`
- `dK = dS Q`
- `dQ = dS^T K`

`lse` is the per-query logsumexp the forward saved, and `delta` is the per-query `rowsum(dO * O)`. Scores,
causal masks and dropout are matched as in the forward. Any of dQ, dK and dV may be absent. Head dims go
up to `MAX_BACKWARD_HEAD` (128). `Backward` records each gradient's writing node and `Access`, the
operand accesses, sizes, scores, causal diagonal, dS scale, dtype and dropout.

## Rewrite

`fuse_backward` replaces each match with either:
- one fusion computing dK, dV and dQ, when the backend's `together` accepts the match and the graph order
  allows it (dQ is then re-matched in its plain form, the dot or one transpose of it), or
- two fusions: one for dK/dV and one for dQ.

Each fusion goes after its last operand is produced and before any of its outputs is read. Extra outputs
leave through `FusionOutput` nodes.

## Backend answers (`AttentionBackwards` trait)

- `mps.rs` - one kernel when `codegen::attention::mps::backward_block_with_dq` finds a block for it and
  `deterministic` is off (dQ is accumulated atomically). Kernel names are requested as
  `Kernel::AttentionBackward`.
- `triton.rs` - always one kernel adding dQ atomically, unless `deterministic`, which gives a dK/dV kernel
  and a dQ kernel so dQ is summed in a fixed order. Kernel name `flash_attention_backward`.
- `tpu.rs` - always a dK/dV kernel and a dQ kernel. `outputs` lists a fusion's dQ, dK and dV accesses,
  and `delta_inside` moves the dot computing `delta` into the fusion body. Both are used by the parent's
  `AttentionLayouts` pass.
- CUDA and CPU: identity.

## Entry points for codegen

`backward_of_body`, `backward_of_body_into` (MPS, outputs written into views) and
`backward_of_body_strided` re-match the backward inside a fusion body.
