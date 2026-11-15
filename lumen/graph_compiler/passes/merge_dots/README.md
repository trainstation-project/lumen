# merge_dots

Merges dots that share an operand into one dot against a packed block. The forward merge turns, e.g.,
the Q, K and V projections of one activation into one matmul against a block of their weights, slicing
each result back out; the backward merge does the same for their gradients' dots. Both run early, right
after `chunked_cross_entropy`: `merge_dots_backward` first, then `merge_dots`. TPU merges later instead,
in the `merge_dots_late` slot after `Simplify` and `dot_strength`.

## Files

- `mod.rs` — re-exports the pass types and declares `MergeDotsLate`, the slot type of the late merge.
- `forward.rs` — `merge_dots`, `MergeDots`, `Target`, `Packs` and the `Blocks` trait.
- `backward.rs` — `MergeDotsBackward`.
- `mps.rs` — MPS: `merge_dots_backward` with `fuse`, `merge_dots` and `contraction_epilogues`;
  `merge_dots` with `fuse` and `merge_dots`, its `Target` reading in place what MPS fusion can fuse, with
  views, lanes 1 and columns `ROW_COLUMNS` (the GEMM epilogue's row columns). Built only when MPS is linked.
- `triton.rs` — Triton: both with `fuse` and `merge_dots`; `Target` from Triton's `fusible`, views,
  lanes 1, 8 columns.
- `cuda.rs` — CUDA: both with `fuse` and `merge_dots`; `Target` from CUDA's `fusible`, views, lanes 1,
  8 columns (16-byte vectors of 16-bit products).
- `tpu.rs` — TPU: `merge_dots_backward` and `merge_dots_late` with `merge_dots`; the late merge's
  `Target` reads everything in place, has no views, packs whole vector lanes (`LANES`), 1 column.

CPU gets the identity pass in all three slots; MPS, Triton and CUDA leave `merge_dots_late` empty, and TPU
leaves the early `merge_dots` slot empty.

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
