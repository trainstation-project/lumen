# `lumen/ops/attention`

Attention kernels. There is no attention primitive: the graph compiler
matches `softmax(q @ kᵀ * scale [masked]) @ v` as traced
(`graph_compiler/attention.rs`, when `CompilerConfig::flash_attention` is on)
and turns it into a fusion whose generated kernel instantiates the templates
here. So this folder has no Rust and no `Op`; it only holds Metal.

## Subfolders

* [`mps/`](mps/README.md) — `forward.metal` (`flash_attention`,
  `attention_decode`) and `backward.metal` (`flash_attention_dkdv`,
  `flash_attention_dq`).

## Related

* `lumen/graph_compiler/attention.rs` — the pattern matcher.
* `lumen/codegen/mps/mod.rs` — `attention_kernel` and the backward's kernel generator.
