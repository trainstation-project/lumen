# `lumen/kernels/dynamic_slice`

Kernels for the `dynamic_slice` and `dynamic_update_slice` primitives: a
block of an operand at start indices that are only known at run time (scalar
tensors, read on the device). Only an MPS backend exists; the module is just
the `mps/` submodule, compiled when `lumen_mps_linked` is set.

Callers: graph plans on MPS reach [`mps::encode`](mps/mod.rs) through the
primitive dispatch in [`lumen/kernels/mps/mod.rs`](../mps/mod.rs). The reference
interpreter (`lumen/kernels/reference.rs`) has its own `DynamicSlice` /
`DynamicUpdateSlice` arms and does not use this folder. The plain device copy
`mps::copy` is reused elsewhere as lumen's MPS buffer-to-buffer copy (see
[`mps/README.md`](mps/README.md)).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder.

## Related

- [`lumen/kernels/mps`](../mps/) — `launch`, `Grid`, arg helpers, shared Metal prelude.
- [`lumen/kernels/scatter_add`](../scatter_add/) — `scatter_add` reuses `mps::copy`.
- `lumen/kernels/reference.rs` — the CPU reference semantics.
