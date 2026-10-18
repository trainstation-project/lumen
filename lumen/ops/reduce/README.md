# `lumen/ops/reduce`

Kernels for the `reduce_sum` and `reduce_max` primitives. Graph primitives,
not `Op`s: on MPS a plan step reaches [`mps/`](mps/README.md) through
`ops::mps::encode` (which also passes scratch for split reductions); off MPS
the plan evaluates it with `ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

* [`mps/`](mps/README.md) — Metal reduction kernels (generic, grouped,
  rows, cols) and the host code choosing among them.

## Related

* `lumen/graph_compiler/mps/` — fused reductions call `mps::encode_reduction`
  with their own kernel names; `fusion_scratch_bytes` uses `mps::scratch_bytes`.
