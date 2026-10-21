# `lumen/ops/factory`

Kernels for the factory primitives, which create a tensor from no tensor
data: `full`, `iota` and `random_bits`. Only an MPS backend exists; the
module is just the `mps/` submodule, compiled when `lumen_mps_linked` is set.

Callers: graph plan steps on MPS reach [`mps::encode`](mps/mod.rs) through
the primitive dispatch in [`lumen/ops/mps/mod.rs`](../mps/mod.rs)
(`Full | Iota | RandomBits`). The reference interpreter
(`lumen/ops/reference.rs`) has its own arms for them. Eager `fill_` /
`zeros` / `ones` on tensors do not come here: they fill on the host and
copy in (`Tensor::fill` in `lumen/tensor/tensor.rs`).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder.

## Related

- [`lumen/ops/mps`](../mps/) — launch helpers, `philox_bits`.
