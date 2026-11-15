# linear_reordering

`LinearReordering` (pass name `linear_reordering`) moves a factor of a matmul's operand to its output:
`f(x) W = f(x W)` for a linear `f` acting on the axes the matmul doesn't contract. A factor constant
along the contracted axis, per lhs row (a norm's `1 / rms`, a softmax's `1 / sum`, a per-token scale) or
per rhs column (a weight's per-channel dequantization scale), then scales the output rows or columns in
the GEMM's epilogue, so the scaled operand is never needed whole. It runs after `ane` and before
`split_k`, so the later fusion passes see the factor after the matmul.

The rounding moves with the factor: a cast between the factor and the matmul now rounds the unscaled
value, so results can differ from the traced program in the last bits. That is why the
`linear_reordering` config flag turns it on, and it is off by default.

## Files

- `mod.rs` — the device-agnostic pass.
- `mps.rs` — MPS runs it with `linear_reordering`, `tile = ROW_TILE` from the MPS GEMM codegen (built only
  when MPS is linked).
- `triton.rs` — Triton runs it with `linear_reordering`, `tile = MOST_GEMM_ROW` from the Triton GEMM
  codegen.

CPU, CUDA and TPU get the identity pass.

## Rewrite

- Matmuls considered (`plain`): unbatched float matmuls of a rank-2+ lhs contracted on its last axis (or a
  2-D lhs on either axis) with a 2-D rhs contracted on one axis.
- `strip` walks an operand back through `Mul`, `Div` by the factor, float `Cast`s and `Neg` to find one
  factor that broadcasts along every axis but the operand's kept ones. It rebuilds the operand without it
  as an `Expr` (each op keeps the origin of the node it rebuilds, so a norm's rows stay in the norm's
  profiler range).
- An operand is stripped only when every reader of it is a matmul reading it on the same side with the
  same kept axes, so the scaled operand is no longer computed at all.
- With `tile = Some(t)`, only a factor that depends on a last-axis reduction over rows wider than `t`
  moves (`crosses_tiles`): a reduction the GEMM's tiles don't hold whole. Every other factor stays as
  traced. `tile = None` moves every factor.
- The new matmul reads the stripped operands and outputs its accumulation dtype; each factor is broadcast
  over the output (`spread`: rows along the leading axes, columns along the last), applied with `Mul` or
  `Div` in the accumulation dtype, and a single `Cast` rounds to the original output dtype.
- The graph is pruned afterwards, dropping the operands nothing reads any more.
