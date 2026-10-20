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

## Related

* `lumen/graph_compiler/attention.rs` — the pattern matcher.
* `lumen/codegen/mps/mod.rs` — `attention_kernel` and the backward's kernel generator.
