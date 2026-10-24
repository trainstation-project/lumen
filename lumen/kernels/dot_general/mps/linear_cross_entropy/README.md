# `lumen/kernels/dot_general/mps/linear_cross_entropy`

`linear_cross_entropy` on MPS: the cross entropy of `h · wᵀ` without writing its logits, in two launches.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `linear_cross_entropy_scratch` (the partials' bytes) and `encode_linear_cross_entropy` (the `lce_partials` launch over tiles of the logits, then `lce_combine`). |
| `linear_cross_entropy.metal` | `LceRows`, the `lce_partials` kernels and `lce_combine`. |
