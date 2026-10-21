# `lumen/ops/attention`

Attention kernels. There is no attention primitive: the graph compiler
matches `softmax(q @ kᵀ * scale [masked]) @ v` as traced
(`graph_compiler/attention.rs`, when `CompilerConfig::flash_attention` is on)
and turns it into a fusion: on MPS its generated kernel instantiates the
Metal templates here; on a TPU it runs the Pallas kernels here, splash
attention's (`jax.experimental.pallas.ops.tpu.splash_attention`), ported.

## Subfolders

* [`mps/`](mps/README.md) — `forward.metal` (`flash_attention`,
  `attention_decode`) and `backward.metal` (`flash_attention_dkdv`,
  `flash_attention_dq`).
* `tpu/` — `forward.py` (splash's `flash_attention_kernel`) and
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
* Blocks are up to 512 queries and 1024 keys (the forward's keys in
  chunks of 512); a causal mask skips the blocks past the diagonal and
  their copies, and masks only the blocks it crosses.
* Not yet: dropout, head dimensions that are not a multiple of 128 when
  heads share rows, an attention whose log-sum-exp is not `[..., S, 1]`.
  These are compile errors naming the operand.

On a v6e (bfloat16, `B=4 S=2048 N=16 H=128`): the forward 717 µs, full
or 702 µs causal (JAX's splash attention 678 / 623 µs); a training step's
attention 2.36 ms full, 2.51 ms causal (splash 2.38 / 2.19 ms). At
`B=2 S=4096 N=32/8`: 2.79 ms forward, 9.14 ms training (splash 2.58 /
9.08 ms); causal 2.52 / 8.24 ms (splash 2.02 / 7.30 ms).

## Related

* `lumen/graph_compiler/attention.rs` — the pattern matcher.
* `lumen/codegen/mps/mod.rs` — `attention_kernel` and the backward's kernel generator.
