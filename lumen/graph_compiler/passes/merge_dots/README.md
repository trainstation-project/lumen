# merge_dots

Merges dots that share an operand into one dot against a packed block. The forward merge turns, e.g.,
the Q, K and V projections of one activation into one matmul against a block of their weights, slicing
each result back out; the backward merge does the same for their gradients' dots. Both run early, right
after `chunked_cross_entropy`: `merge_dots_backward` first, then `merge_dots`. TPU's list merges forward dots
later instead, after `Simplify` and `dot_strength`.

## Files

- `mod.rs` — re-exports the pass types.
- `forward.rs` — `merge_dots`, `MergeDots` (`configured_with(config, target)`: with `merge_dots`, packing for the
  device's `Target`), `Target`, `Packs` and the `Blocks` trait.
- `backward.rs` — `MergeDotsBackward` (with `merge_dots`);
  and the backward side of the forward merge's interleaving: `interleaved_gradients` (the gradient concatenates
  and slices to rewrite, an `Interleaving`) and `arrange` (a gradient concatenation in the interleaved layout).

The code is the same on every device; each device's pass list (`graph_compiler/pipeline/<device>.rs`) gives its
`Target`, where its forward merge runs, and its gates beyond `merge_dots`:

| Device | Gates | Forward merge | `Target` |
|---|---|---|---|
| MPS | `fuse`; backward also `contraction_epilogues` | early | what MPS fusion fuses, views, lanes 1, `ROW_COLUMNS` columns (the GEMM epilogue's row columns) |
| Triton | `fuse` | early | Triton's `fusible`, views, lanes 1, 8 columns |
| CUDA | `fuse` | early | CUDA's `fusible`, views, lanes 1, 8 columns (16-byte vectors of 16-bit products) |
| TPU | none | late | everything read in place, no views, `LANES` lanes (whole vector lanes), 1 column |

The CPU merges nothing.

## Forward merge

`merge_dots(graph, packable, donated, target)` finds dots that share one operand and whose other operand
derives from a packable graph input (`Options::packable`), optionally through a single-use elementwise
chain that is replayed on the block. The result must be read only in place (by dots or nodes
`target.reads` accepts), so slicing it is free.

- The weights are concatenated along the last free dim of the non-shared operand; the merged dot reads a
  new graph input, the packed block, and each member's result is sliced back out.
- A block concatenated along a non-leading dim stores each weight strided, so a donated weight or one read
  elsewhere can't join. Without `views`, a dot joins only when the shared operand is its lhs, the weight is
  stored unstrided, and the weight operand's two trailing dims are multiples of `lanes`.
- A dot's rounding cast merges with it: the merged dot is rounded by one cast.
- Members whose results are combined elementwise may be interleaved in regions aligned to `columns`; a
  group that only partly interleaves splits into the interleaved part and the rest. The backward graph's
  gradient concatenates and slices are rewritten to follow that layout.
- A group with the same weights as an earlier block (e.g. its backward) reuses that block.

`MergeDots` runs on a module that implements `Blocks`, and stores the returned packs (per block, the graph
input positions it concatenates and the dim) and interleaved region sizes on it. Packed blocks are
appended as the graph's last inputs; `graph_compiler::plan_options` treats them as parameters.

## Backward merge

`MergeDotsBackward` finds a sum of same-shaped dots `g_i . a_i` over weights `a_i` that share a forward
input and rewrites it as one dot of the concatenated cotangents and weights. The weight-gradient dots
reading those cotangents become one dot sliced per weight. Transposes applied to every summed dot are
looked through when they don't reorder free dims, and rounding casts merge as in the forward merge.
