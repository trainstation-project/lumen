# `lumen/kernels/scatter_add`

Kernels for the `scatter_add` primitive (add along one `axis` at an
integer index tensor's positions). `gather` has no kernel: it is generated
(in a fusion, or alone as a one-node fusion, `graph_compiler/mps/alone.rs`). Only an MPS backend exists; the module
is just the `mps/` submodule, compiled when `lumen_mps_linked` is set.

Callers: graph plan steps on MPS reach [`mps::encode`](mps/mod.rs) through
the primitive dispatch in [`lumen/kernels/mps/mod.rs`](../mps/mod.rs)
(`ScatterAdd { .. }`). The reference interpreter
(`lumen/kernels/reference.rs`) has its own `ScatterAdd` arm.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder.

## Related

- [`lumen/kernels/dynamic_slice`](../dynamic_slice/) — runtime-offset block access; its `copy` is used by `scatter_add`.
