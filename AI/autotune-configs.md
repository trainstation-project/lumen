---
name: autotune-configs
description: Use whenever adding or changing an autotune candidate list, a kernel's untuned default, or a block/tile/lane size a kernel launches with, on any device. Says where those values live.
---

# Autotune configs live in `config.rs`

Every value that picks how a kernel launches goes in the `config.rs` next to
the code that uses it, never inline in codegen, kernels or passes:

- the candidates the autotuner tries (tiles, block sizes, warps, stages);
- the config used when autotuning is off;
- block, tile and lane sizes, and the thresholds that choose between them
  (split a reduction below N outputs, decode kernel at 8 queries or fewer);
- which steps are worth tuning (MPS's `TUNED_ELEMENTS`).

## Where

| Backend | Kernels | File |
|---|---|---|
| Triton | GEMM, loops, reductions, flash attention | `lumen/codegen/triton/config.rs` |
| MPS | loop tunings, tuned sizes, flash-attention blocks | `lumen/codegen/mps/config.rs` |
| MPS | GEMM tiles, one file per algorithm | `lumen/kernels/dot_general/mps/{tiled,skinny,gemv,mpp}/config.rs` |
| TPU | flash-attention blocks | `lumen/kernels/attention/tpu/config.rs` |

A kernel family without one gets a `config.rs` in its own module, declared
with `mod config;` and its items re-exported or imported where used.

## How

- Name each value; give it a one-line doc comment saying what it bounds or
  why that number (`/// Reductions with fewer outputs than this are split
  across programs to fill the GPU.`).
- Candidate lists are `const` arrays (or a `OnceLock` when generated),
  largest tiles first, with the untuned pick named as an index or function.
- Keep the narrowest visibility that works: `pub(super)` when only the
  sibling codegen reads it.
- Keep the code that uses a value free of literals for it: it reads the
  named constant or calls the config function.
- Don't move code that derives a size from the hardware (VMEM budgets,
  TPU tile layouts) unless it is a tunable choice.
