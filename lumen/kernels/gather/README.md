# `lumen/kernels/gather`

Kernels for the `gather` and `scatter_add` primitives (index along one
`axis` with an integer index tensor). Only an MPS backend exists; the module
is just the `mps/` submodule, compiled when `lumen_mps_linked` is set.

Callers: graph plan steps on MPS reach [`mps::encode`](mps/mod.rs) through
the primitive dispatch in [`lumen/kernels/mps/mod.rs`](../mps/mod.rs)
(`Gather { .. } | ScatterAdd { .. }`). The reference interpreter
(`lumen/kernels/reference.rs`) has its own `Gather` / `ScatterAdd` arms.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder.

## Related

- [`lumen/kernels/dynamic_slice`](../dynamic_slice/) — runtime-offset block access; its `copy` is used by `scatter_add`.
- [`lumen/kernels/layout`](../layout/) — note its `gather_<bytes>` kernel is a strided copy, not this primitive.
