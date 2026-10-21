# `lumen/ops/reduce`

Kernels for the `reduce_sum` and `reduce_max` primitives. Graph primitives,
not `Op`s: on MPS a plan step reaches [`mps/`](mps/README.md) through
`ops::mps::encode` (which also passes scratch for split reductions); on a TPU,
every reduction is a `reduction` fusion (alone, or with its prologue and
epilogue), whose generated spec (`codegen/tpu/reduction.rs`) runs
[`tpu/`](tpu/)'s template; elsewhere the plan evaluates it with
`ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `mps` / `tpu` hooks (`#[cfg(lumen_mps_linked)]`, `#[cfg(lumen_tpu_linked)]`). |
| `tpu/mod.rs` | The TPU template's schedule: `blocks` for each layout (rows, columns, total), the reduced axes stepped over into an accumulator. |
| `tpu/__init__.py` | The template, in lumen's Pallas: `reduce_sum`/`reduce_max` over the last axis (`[m, n]` to `[m]`, `[lead, h, m, n]` to `[lead, h, m]`), every axis but the last (to `[n]`) or every axis (to a scalar), an operand block reduced into a VMEM accumulator a step; the spec's prologue replayed on operand blocks, its epilogue on the finished results. |

## Subfolders

* [`mps/`](mps/README.md) — Metal reduction kernels (generic, grouped,
  rows, cols) and the host code choosing among them.

## Related

* `lumen/graph_compiler/mps/` — fused reductions call `mps::encode_reduction`
  with their own kernel names; `fusion_scratch_bytes` uses `mps::scratch_bytes`.
