# `lumen/graph_compiler/passes`

The device-independent compiler passes: rewrites of the graph alone, with no
knowledge of any device's kernels, tiles or memory. Each is a
[`CompilerPass`](../pass.rs) implemented for **any** `M: Module`, so the
frontend pipeline, the CPU's and the MPS's can all run it. A device's own
passes (those that request its kernels, follow its matmul layouts or tile
sizes, or fuse for it) live in that device's folder, e.g.
[`../mps/`](../mps/README.md).

## Files

| File | Pass (`name()`) | Contents |
| --- | --- | --- |
| `mod.rs` | — | Declares the passes. |
| `cse.rs` | `Cse` (`cse`) | Common subexpression elimination (XLA's `HloCSE`): a node applying the same primitive, parameters included, to the same values as an earlier one is that node's value; commutative primitives match in either operand order. |
| `simplify.rs` | `Simplify { matched }` (`simplify`) | Algebraic simplification (`x * 1`, `x + 0`, composed transposes, ...). With `matched` (after a device's patterns are matched), also swaps transposes of dots and moves reshapes, transposes and casts; the frontend runs it unmatched so devices can match patterns first. |
| `gated_backward.rs` | `GatedBackward` (`gated_backward`) | A gated pair's or group's backward (SwiGLU's gate/up, attention's q/k/v): its input gradient's sum of dots one dot of `[g1 \| g2 \| ...]` and the weights' block, its weights' gradients one dot. Only where the other operands are weights. |
| `linear_cross_entropy.rs` | `LinearCrossEntropy` (`linear_cross_entropy`) | A cross entropy of a dot's logits nothing else reads, as written, becomes `Primitive::LinearCrossEntropy` (Cut Cross-Entropy): the logits never reach memory. Also the pattern matcher (`matched`, `uses`, `Found`) the chunked pass reuses. |
| `chunked_cross_entropy.rs` | `ChunkedCrossEntropy` (`chunked_cross_entropy`) | A linear cross entropy trained a chunk of rows at a time (`fused_linear_cross_entropy_chunk_size`): its losses computed from the chunks' logits its gradient computes, each chunk's logits once (XMA's fused linear cross entropy). |
| `dot_strength.rs` | `DotStrength { matrices }` (`dot_strength`) | A dot of two vectors (no free dimensions: `rowsum(dO * O)`) becomes a product and a `reduce_sum` (XLA's `DotStrengthReduction`); with `matrices` a dot of a vector and a matrix too (the MPS's; the TPU's dot kernel streams the matrix at nearly its bandwidth, and runs the vector·vector ones only). |
| `take_along_axis.rs` | `TakeAlongAxis` (`take_along_axis`) | Each `take_along_axis` a `Gather` of its operand flattened, at each element's flat position (iotas and the indices times strides): for a device that gathers along one axis only (the MPS). |
| `fold_dot_reshapes.rs` | `FoldDotReshapes` (`fold_dot_reshapes`) | A dot of a reshape merging its lhs's leading dimensions a dot of the lhs itself, reshaped back (a TPU's, where those reshapes are copies). |
| `clamp_indices.rs` | `ClampIndices` (`clamp_indices`) | With `safe_kernels`: gather, scatter-add and linear cross entropy indices clamped into range before the kernels read them (XLA's gather semantics). |

## Writing a pass

```rust
pub(crate) struct MyPass;

impl<M: Module> CompilerPass<M> for MyPass {
    fn name(&self) -> &'static str {
        "my_pass" // what the profiler range and dump_to file are called
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = my_rewrite(m.graph());
        Ok(())
    }
}
```

Keep the rewrite itself a plain `fn(&Graph) -> Graph` beside the pass: the
unit tests call those directly. Then add the pass to the pipelines that
should run it (`Pipeline::add`): `graph_compiler/mod.rs` (frontend),
`cpu/mod.rs`, `mps/mod.rs`, `tpu/mod.rs`.

## Invariants & gotchas

- **No device knowledge here.** If a rewrite needs a device's kernel limits,
  tiles, layouts or `MpsModule` state, it belongs in that device's folder.
  These files import only the graph IR and the shared helpers in
  `graph_compiler/mod.rs` (`elementwise`, `constant`).
- **Pass order is the pipeline's, not this folder's.** The MPS pipeline runs
  `chunked_cross_entropy` before `gated_backward` before `merge_dots`, and
  `linear_cross_entropy` after flash attention but before `clamp_indices`;
  the comments in `mps/mod.rs`'s `pipeline()` say why.
- **Rewrites rebuild the graph.** Keep each node's origin
  (`set_origin`/`set_origins`) and label when copying or replacing nodes, so
  profiles and dumps still point at the traced program's lines.
- **Names are load-bearing:** `name()` is the `dump_to` file suffix and the
  profiler range (`<pipeline>::<pass>`); tests check some of them.

## Related

- [`../`](../README.md) — `pass.rs` (`CompilerPass`, `Module`, `Pipeline`), the frontend pipeline.
- [`../mps/`](../mps/README.md) — the MPS pipeline (which runs most of these) and its device-specific passes.
- [`../cpu/`](../cpu/README.md) — the CPU pipeline (`simplify`).
- [`../../graph/`](../../graph/README.md) — the IR these passes rewrite.
