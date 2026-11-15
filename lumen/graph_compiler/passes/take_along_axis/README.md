# take_along_axis

`TakeAlongAxis` (pass name `take_along_axis`) lowers each `Primitive::TakeAlongAxis` to a flat `Gather`,
so no backend needs a kernel of its own for it. It is the second pass in the list, right after
`unsupported`.

## Files

- `mod.rs` — the device-agnostic pass.
- `mps.rs` — MPS runs it always (built only when MPS is linked).
- `triton.rs` — Triton runs it always.

CPU, CUDA and TPU get the identity pass.

## Rewrite

For `take_along_axis(x, indices, axis)`:

1. The index dtype is int64 when the indices are 64-bit (`I64` or `U64`), int32 otherwise; the indices are
   cast to it when needed.
2. A flat position of the indices' shape is built as the sum over `x`'s dimensions of the coordinate times
   the row-major stride: the (cast) indices along `axis`, an `Iota` along every other dimension, each
   multiplied by a `Full` of the stride.
3. `x` is reshaped to one dimension of `x.numel()` elements and gathered with `Gather { axis: 0 }` at the
   flat positions.

The gather keeps the original node's label; every other node is copied unchanged.
