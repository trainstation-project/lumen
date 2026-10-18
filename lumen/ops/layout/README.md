# `lumen/ops/layout`

Kernels for the layout primitives in a plan: `reshape` (when the plan
copies), `broadcast_in_dim`, `transpose` and `slice`. Only an MPS backend
exists; the module is just the `mps/` submodule, compiled when
`lumen_mps_linked` is set.

Callers: graph plan steps on MPS reach [`mps::encode`](mps/mod.rs) through
the primitive dispatch in [`lumen/ops/mps/mod.rs`](../mps/mod.rs)
(`Reshape | BroadcastInDim | Transpose | Slice`). The reference interpreter
(`lumen/ops/reference.rs`) handles these primitives itself.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder.

## Related

- [`lumen/ops/mps`](../mps/) — launch helpers and `offset_of32`.
- [`lumen/ops/gather`](../gather/) — the indexed `gather` primitive (different kernels).
