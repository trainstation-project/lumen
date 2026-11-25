# passes/place_concats

Concatenations written in place: `place_concats`, before fusion, marks concatenations whose operands' producers can
write straight into their slots; `concats`, after it, copies the ones fusion left unable to be.

## Files

- `mod.rs` - `in_place_concats`, `worth`, the `Slots` trait (which producers write a slot, and which slots are
  worth placing), the `Rechecks` trait (the rule after fusion) and the `PlaceConcats` and `Concats` passes.
- `mps.rs` - MPS's `Slots` and `Rechecks`: what its kernels can write through a slot's strides.
- `triton.rs` - Triton's `Slots`: a kernel of its own writing fresh memory (a matmul, or a fusible elementwise or
  copying layout op).
- `tpu.rs` - TPU's `Slots` and `Rechecks`, and `slot_writes`, which `dot/fuse` reads too.
