# `lumen/ops/attention`

Attention kernels. There is no attention primitive: the graph compiler
matches `softmax(q @ kᵀ * scale [masked]) @ v` as traced
(`graph_compiler/common/attention.rs`, when `CompilerConfig::flash_attention` is on)
and turns it into a fusion: on MPS its generated kernel instantiates the
Metal templates here; on a TPU it runs the Pallas kernels here, splash
attention's (`jax.experimental.pallas.ops.tpu.splash_attention`), ported.

## Subfolders

* [`mps/`](mps/README.md) — `forward.metal` (`flash_attention`,
  `attention_decode`) and `backward.metal` (`flash_attention_dkdv`,
  `flash_attention_dq`).
* `tpu/` — the templates (its specs generated in `codegen/tpu/attention.rs`; `mod.rs` the blocks and launch): `forward.py` (splash's `flash_attention_kernel`) and
  `backward.py` (its dq and dkv kernels), in lumen's Pallas; `__init__.py`
  their shared parts (each head's slab and column of its operand, the
  scores' roundings and scale, the causal mask, the blocks it skips);
  `mod.rs` the spec each runs on (`forward`, `backward`: what the matcher
  found, each operand read where the program wrote it) and `encode`. A
  causal mask skips the blocks past the diagonal and clamps their copies;
  a block of keys' dk and dv sum over the query heads of its group
  (grouped-query attention), written once its last head is done.

## TPU

The graph compiler's TPU pipeline fuses attention (`flash_attention`,
`flash_attention_backward` in `graph_compiler/tpu`) with the shared
matcher, then lays the fusions' operands out for the kernels:

* Each operand is read where the program wrote it: per head, a slab of
  its buffer and a column block of `head_dim` (a multiple of 128, or the
  whole row). `[B, S, N, H]` operands are read through the `[B, S, N*H]`
  reshape they came from, never copied.
* A matrix output (o, dq, dk, dv) is written as the reshapes, casts and
  group sums after it would leave it (`reshaped_outputs`): grouped-query
  dk and dv summed over each group in the dkv kernel. One whose rows are
  not its tiles' rows and that nothing reshapes is written with its last
  two dimensions merged (`merged_outputs`), reshaped back for its readers.
* The backward's `delta = rowsum(dO o)` is computed in its kernels from o
  and dO (`delta_inside`), not as separate kernels.
* Blocks are up to 1024 queries and 4096 keys, computed in chunks of 512
  keys: each grid step of the in-kernel pipeline costs about 0.3 us of
  scalar work whatever it computes, so fewer, larger steps win. A causal
  mask skips the chunks past the diagonal (and the copies of whole
  blocks past it, fetching the next row's first block instead), masking
  only the chunks it crosses.
* Not yet: dropout, head dimensions that are not a multiple of 128 when
  heads share rows, an attention whose log-sum-exp is not `[..., S, 1]`.
  These are compile errors naming the operand.

On a v6e, bfloat16, the scale folded into q as JAX's splash attention
takes it (its blocks 512 by 1024, chunks of 512):

| B, S, heads | | forward | splash | training step | splash |
| --- | --- | --- | --- | --- | --- |
| 4, 2048, 16 | full | 469 us | 678 us | 1.75 ms | 2.38 ms |
| 4, 2048, 16 | causal | 592 us | 623 us | 2.19 ms | 2.19 ms |
| 2, 4096, 32/8 | full | 1.71 ms | 2.58 ms | 6.40 ms | 9.08 ms |
| 2, 4096, 32/8 | causal | 1.74 ms | 2.02 ms | 6.36 ms | 7.30 ms |

With the scale applied to the scores, as `F.flash_attention` traces it,
each costs about 3 to 10% more.

## Related

* `lumen/graph_compiler/common/attention.rs` — the pattern matcher.
* `lumen/codegen/mps/mod.rs` — `attention_kernel` and the backward's kernel generator.
