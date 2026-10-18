# `lumen/ops/dot_general`

Kernels for the `dot_general` primitive (XLA's batched, generalized
matmul) and for `linear_cross_entropy` (the logits `h · wᵀ` fused with a
cross-entropy, never materialized). There is no `Op` here: `dot_general` is
a graph primitive, run by a compiled plan. On MPS, `ops::mps::encode` sends
its steps to `mps::encode` / `mps::encode_linear_cross_entropy`; off MPS the
plan evaluates it with `ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

* [`mps/`](mps/README.md) — the Metal matmul kernels (integer, 64-bit
  integer, simdgroup-matrix float, split-K atomic, linear-cross-entropy) and
  the host code that canonicalizes a `dot_general` to `[b, m, k] × [b, k, n]`
  and picks a kernel and tile.

## Related

* `lumen/graph_compiler/mps/canonicalize_dots.rs`, `split_k.rs`,
  `place_views.rs` — compiler passes that use this folder's matmul-form
  helpers.
* `lumen/codegen/mps/mod.rs` — fusion kernels that call the `matmul_sg_impl` template.
