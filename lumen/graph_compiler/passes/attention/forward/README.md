# passes/attention/forward

The forward flash-attention pass (`FlashAttention`, named `flash_attention`). It finds each live attention
`softmax(scores(Q K^T)) V` in the graph and replaces it with one `flash_attention` fusion. It is the first
attention pass in the device pipeline and runs right before `flash_attention_backward`.

## Pattern

The match starts from the second dot (`P V`) and walks back:
- `P` is `exp(s - max) / sum`, optionally rounded by a cast and multiplied by a dropout mask.
- `s` is `Q K^T` followed by `Score` ops (casts and constant scales).
- An optional causal mask keeps key `k` for query `q` when `k <= q + causal`, plus an optional runtime
  offset read from a rank-0 integer input.
- `max + log(sum)` becomes an extra output (`lse`) when the graph also reads the logsumexp. The backward
  pass reads it.
- The softmax input is float32, and both dots and the sum accumulate in float32; other dtypes do not match.
- Q, K, V and the output are recorded as strided `Access`es, with head dims up to `MAX_HEAD` (256).

`Attention` holds the match: accesses, batch dims, sizes, scores, causal diagonal and offset, the two dot
nodes, dtypes and dropout.

## Rewrite

`fuse` replaces each match the backend `supports` with a `Fusion` whose body is the subgraph from the
operands to the output. Extra outputs (the raw attention when also read, the lse) leave through
`FusionOutput` nodes. When the backend allows it, the elementwise epilogue chain after the attention
is fused too; the epilogue is dropped if the raw attention or the lse is read before the chain ends.
A runtime causal offset must be a buffer, so a match whose offset is a by-value scalar is skipped.

## Backend answers (`Attentions` trait)

- `mps.rs` - epilogues when `contraction_epilogues` is on; by-value scalars from the module; kernel
  names requested as `Kernel::Attention`; runtime causal offsets allowed.
- `triton.rs` - epilogues always on, reading any leaf; by-value scalars from `Options::scalars`; kernel
  name `flash_attention`; runtime offsets allowed (decode).
- `cuda.rs` - no epilogues, no by-value scalars, static causal diagonal only, and only the attentions
  `codegen::attention::cuda::supports` accepts. The rest stay unfused.
- `tpu.rs` - epilogues when `contraction_epilogues` is on, reading only scalar leaves, through reshapes
  but not transposes; no by-value scalars; static causal diagonal. `outputs` gives a fusion's output
  `Access` to the parent folder's `AttentionLayouts` pass.

## Other entry points

- `attentions(graph, runtime_offsets)` lists every live match.
- `traced(graph)` (Python feature) reports each attention found while tracing, with its output and the
  values between operands and output.
- `of_body` and `of_body_strided` re-match an attention inside a fusion body for codegen.
