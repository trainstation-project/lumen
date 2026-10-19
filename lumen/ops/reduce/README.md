# `lumen/ops/reduce`

Kernels for the `reduce_sum` and `reduce_max` primitives. Graph primitives,
not `Op`s: on MPS a plan step reaches [`mps/`](mps/README.md) through
`ops::mps::encode` (which also passes scratch for split reductions); on a TPU,
a `reduce_sum` over the last dimension reaches [`tpu/`](tpu/) through
`ops::tpu::encode`; elsewhere the plan evaluates it with `ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `mps` / `tpu` hooks (`#[cfg(lumen_mps_linked)]`, `#[cfg(lumen_tpu_linked)]`). |
| `tpu/mod.rs` | The TPU `reduce_sum`'s spec: a sum over the last dimension of two or more, its blocks (the summed one's stepped over into an accumulator; whole if no lane block divides it). |
| `tpu/__init__.py` | Its kernel, in lumen's Pallas: `[m, n]` to `[m]`, or `[lead, h, m, n]` to `[lead, h, m]`, an operand block's rows summed into a VMEM accumulator a step. |

## Subfolders

* [`mps/`](mps/README.md) — Metal reduction kernels (generic, grouped,
  rows, cols) and the host code choosing among them.

## Related

* `lumen/graph_compiler/mps/` — fused reductions call `mps::encode_reduction`
  with their own kernel names; `fusion_scratch_bytes` uses `mps::scratch_bytes`.
