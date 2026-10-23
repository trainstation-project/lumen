# `lumen/graph_compiler/common`

What every device's compiler shares (XLA's `hlo/transforms`, beside each
backend's own): the device-independent compiler passes, rewrites of the
graph alone, with no knowledge of any device's kernels, tiles or memory;
and the matchers a device specializes through a trait or a filter
(attention, normalization diamonds, horizontal fusion). Each pass is a
[`CompilerPass`](../pass.rs) implemented for **any** `M: Module` (or one
with the trait it asks, as `Blocks` or `Attentions`), so the frontend
pipeline and the CPU's, MPS's and TPU's can all run it. A device's own
passes (those that request its kernels, follow its matmul layouts or tile
sizes, or fuse for it) live in that device's folder, e.g.
[`../mps/`](../mps/README.md).

## Files

| File | Pass (`name()`) | Contents |
| --- | --- | --- |
| `mod.rs` | — | Declares the passes and matchers. |
| `horizontal.rs` | — | Horizontal fusion (XLA's horizontal loop fusion): independent fusions grouped into one multi-output fusion (`fuse`), what a device groups and how much its `Target` says (MPS: `../mps/horizontal.rs`, TPU: `../tpu/fusion/horizontal.rs`). |
| `diamonds.rs` | — | Normalization diamonds (XLA `SoftmaxRewriterTriton`): `Row`s of a row's reductions and the elementwise work around them, chained and grown (`diamonds`, over a device's `Fusible`); MPS and TPU each turn them into row kernels. |
| `attention.rs` | — | Pattern matcher for attention (`softmax(q @ k^T * scale [mask]) @ v`, dropout, its backward), shared by the MPS fusions, the codegen and the tracer (`Graph.attentions`). |
| `autotune.rs` | — | Compile-time autotuning, after xma's `autotune`. `Autotuner { timer }.tune(kernels, plan)`, the one driver: for each step a family has configs for, build a kernel per config, compile them together, time each on a copy of the step carrying it (one warm run sets the repeats to about 1 ms, best of two passes), and set the fastest as the step's `tuned` (`plan.rs`'s `Tuned`: the kernel and its config), once for the steps of the same fusion or primitive and types. `Kernels`, a kernel family (`configs` for a step, its search space, empty when not tuned; `build` one config's kernel; `compile` those built). `Timer`, a device's (`buffers` for a step; `time` it for some repeats). Encode reads the pick with `Step::tuned::<Config>()`. |
| `loops.rs` | — | Loop fusions from a device's grouping (each node's group, its root): `members`, `body` (a group as a one-output graph and the values it reads) and `fused` (the graph with each group one `Fusion`, named and placed by the device), for the TPU's `loop_fusion` and the Triton backend's. |
| `cse.rs` | `Cse` (`cse`) | Common subexpression elimination (XLA's `HloCSE`): a node applying the same primitive, parameters included, to the same values as an earlier one is that node's value; commutative primitives match in either operand order. |
| `simplify.rs` | `Simplify { matched }` (`simplify`) | Algebraic simplification (`x * 1`, `x + 0`, composed transposes, ...). With `matched` (after a device's patterns are matched), also swaps transposes of dots and moves reshapes, transposes and casts; the frontend runs it unmatched so devices can match patterns first. |
| `merge_dots.rs` | `MergeDots { target }` (`merge_dots`) | Dots sharing an operand become one dot over a packed block of parameters (XLA `DotMerger`), for a module with `Blocks` (its `packed` and `interleaved`). `Target` is the device's say: `reads` (which readers read a merged result), `views` (whether members may be views of a block: strided weights and interleaved groups, MPS), `lanes` (members' sizes along their last two dimensions multiples of it: 128 on a TPU) and `columns` (an interleaved row's width: the MPS's 8). Groups of K dividing `columns` get an interleaved region each (`Plan::interleaved`); otherwise side by side. On a TPU (`views` off) only weights side by side along their rows (`nn.Linear`'s `[out, in]`) merge. |
| `flash_attention.rs` | `FlashAttention` (`flash_attention`) | Each attention one fusion (`attention::fuse`), for a module with `Attentions`, the device's say: its kernel's name, the runtime scalars, and with `contraction_epilogues` which epilogues it takes (`Epilogues`: the leaves its kernel reads, every one on MPS, scalars on a TPU; `reshaped`, a TPU's, keeps the raw output after the chain's leading reshapes, so heads are merged, and leaves a chain of only reshapes to the output's layout). |
| `merge_dots_backward.rs` | `MergeDotsBackward` (`merge_dots_backward`) | A gated pair's or group's backward (SwiGLU's gate/up, attention's q/k/v): its input gradient's sum of dots one dot of `[g1 \| g2 \| ...]` and the weights' block, its weights' gradients one dot. Only where the other operands are weights. |
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
  `chunked_cross_entropy` before `merge_dots_backward` before `merge_dots`, and
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
