# fold_reshapes

`FoldDotReshapes` (pass name `fold_dot_reshapes`) lets a matmul read a rank-3+ activation directly
instead of through a reshape that flattens it to a matrix. It runs after `linear_cross_entropy` and
just before the matched `Simplify`, early in the one pass list.

## Files

- `mod.rs` — the device-agnostic pass.
- `tpu.rs` — TPU runs it always.

Every other device (CPU, MPS, CUDA, Triton) gets the identity pass.

## Rewrite

It matches `dot(reshape(x), w)` where:

- the dot is unbatched and contracts dimension 1 of a 2-D lhs;
- the lhs is made by a `Reshape` of `x`, `x` has rank above 2, and `x`'s last dimension equals the lhs's
  last dimension (the reshape only merges leading dimensions).

It emits `reshape(dot(x, w))`: the new dot contracts `x`'s last dimension, keeps the original rhs
contracting/batch dims and the accumulation and output dtypes, and the reshape restores the original
2-D output shape. The pass copies the old reshape node as it is and does not remove it.

Nodes it doesn't rewrite are copied with their labels and origins.
