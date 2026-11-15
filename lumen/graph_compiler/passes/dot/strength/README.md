# strength

`DotStrength` (pass name `dot_strength`) rewrites dots between vectors (batched inner products, such as
attention backward's delta read through transposes) as an elementwise multiply and a reduce-sum, so they
run as elementwise/reduction fusions instead of a matmul kernel. It runs right after the matched
`Simplify` and before `merge_dots_late`.

## Files

- `mod.rs` — the device-agnostic pass and its helpers (`product`, `cast_with`, `reshaped`).
- `mps.rs` — MPS runs it always (built only when MPS is linked).
- `triton.rs` — Triton runs it always.
- `tpu.rs` — TPU runs it always.

CPU and CUDA get the identity pass.

## Rewrite

It matches a `DotGeneral` whose accumulation dtype is a float and where both operands have only size-1
free dimensions (every non-batch, non-contracting dim multiplies to 1).

It emits, in order:

1. the first operand reshaped to drop its size-1 free dims and transposed into the second operand's
   dimension order;
2. casts of both operands to the accumulation dtype, when they differ from it;
3. a `BroadcastInDim` of the first operand onto the second's shape, when the shapes differ;
4. `Mul`, then `ReduceSum` over the second operand's contracting axes in the accumulation dtype;
5. a transpose to dot output order (batch dims, then free dims) when needed, a reshape to the dot's output
   shape, and a cast to the output dtype when it differs from the accumulator.

Integer dots and dots with any non-trivial free dimension are copied unchanged.
