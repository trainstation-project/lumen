# graph_compiler/passes

Every graph rewrite the compiler runs between tracing and codegen. There is one device-agnostic pass
list (`pipeline::pipeline!`); each slot holds a pass, and each device runs either its own version of that
pass or the identity pass, which the pipeline skips.

## How a pass is written

- `Module` is what a pass rewrites: a graph, its `Options` and the device's `CompilerConfig`.
- `CompilerPass<M>` has `name` (used for profiler ranges and dump files), `apply` and `is_empty`.
- A device runs a pass by implementing `DevicePass<D>` for the pass's slot type in
  `passes/<pass>/<device>.rs`. That impl returns the pass, usually behind config flags via
  `pipeline::when`. Without such an impl, `pick!` returns `identity::Identity`.
- Shared matching and rewriting live in the pass's `mod.rs`. Device answers (what is fusible, kernel
  names, limits) come in through a trait the pass defines and `DeviceModule<D>` implements.
- A pass whose devices differ only in a value their codegen knows is one file, `passes/<pass>.rs`: a generic
  `impl<D: Trait> DevicePass<D>` reads it through a small trait each backend's codegen implements, so every
  device implementing the trait runs the same pass (`linear_reordering.rs`, `RowTile`).

## Frontend (every device, before staging)

1. `preserve_precision` - drop `cast(cast(x, b), a)` round trips for forward readers (flag
   `preserve_precision`).
2. `cse` - common subexpression elimination (never of custom calls or collectives: each collective is a message every process must issue).
3. `simplify` (`matched: false`) - algebraic simplification without structural rewrites.

## Device pipeline, in order (devices that run it in brackets)

1. `unsupported` - reject ops the device has no kernel for [mps, triton].
2. `take_along_axis` - lower `take_along_axis` to a flat `gather` [mps, triton].
3. `chunked_cross_entropy` - replay a cross entropy per row chunk when the logits are already chunked [mps, triton].
4. `merge_dots_backward` - merge gradient dots that share an operand [cuda, mps, triton, tpu].
5. `merge_dots` - merge forward dots sharing an operand into one dot against a packed weight block [cuda, mps, triton].
6. `flash_attention` - replace `softmax(QK^T)V` with a `flash_attention` fusion [cuda, mps, triton, tpu].
7. `flash_attention_backward` - replace attention gradients with backward fusions [mps, triton, tpu].
8. `attention_layouts` - fold attention output layout chains and `delta` into the fusions [tpu].
9. `linear_cross_entropy` - replace `cross_entropy(h @ w, target)` with one `LinearCrossEntropy` op [mps, triton].
10. `fold_dot_reshapes` - fold a reshape flattening leading dims into a 2-D dot's lhs [tpu].
11. `simplify` (`matched: true`) - full simplification, including structural rewrites [every device, CPU too].
12. `swap_pairs` - pair swaps written as slices and a concatenation become `SwapPairs` (an interleaved rotary's partners, read at `j ^ 1` or swapped in a GEMM accumulator's registers) [mps, triton].
13. `dot_strength` - vector-vector dots become multiply plus reduce-sum [mps, triton, tpu].
14. `merge_dots` (`MergeDotsLate`) - dot merging after simplification, packing whole vector lanes [tpu].
15. `neural_engine` - offload float16 GEMM regions over fixed weights to the Apple Neural Engine [mps].
16. `linear_reordering` - move a per-row or per-column factor of a matmul operand to its output [mps, triton].
17. `split_k` - split long-K, few-tile matmuls into partial products over K chunks [mps].
18. `canonicalize_dots` - transpose dot operands into the layout the matmul kernels read [mps, tpu].
19. `normalization_diamonds` (`FusedDiamonds`) - replace normalization rows with `row` fusions [tpu].
20. `dot_rows` - fuse a matmul into the `row` fusion that reads its whole rows [tpu].
21. `dynamic_update_slices` (`Updates`) - chains of `dynamic_update_slice` become one in-place fusion [mps, triton, tpu].
22. `normalization_diamonds` (`Diamonds`) - find normalization rows and keep them for fusion [mps, triton].
23. `fuse_dots` - fuse TPU matmuls with their slices, stacked lhs and epilogues [tpu].
24. `place_concats` - let producers write directly into their concatenation slot [mps, triton, tpu].
25. `fusion` - group elementwise chains, reductions, rows and epilogues into `Fusion` kernels [cuda, mps, triton, tpu].
26. `wide_rows` - GEMMs feeding rows wider than a tile also write the row's partial reductions [triton].
27. `place_views` - slices read only by GEMMs become strided views [mps, triton].
28. `alone` - wrap leftover nodes that only a generated kernel can compute into one-node fusions [cuda, mps, triton].
29. `concats` - copy the concatenations fusion could not place in place [mps].
30. `horizontal_fusion` - merge independent small kernels of the same kind into one launch [triton, tpu].

## Other files

- `mod.rs` - the `Module`, `DevicePass` and `CompilerPass` traits.
- `identity.rs` - the identity pass.
- `elementwise.rs` - building elementwise fusion bodies from node groups (used by fusion).
- `multi_output.rs` - `Readers`: a fusion also writes a value other fusions read as an extra output.
- `rms_norm.rs` - `rms_norms` detects RMS norms written as primitives, so autograd can differentiate
  them from saved `1 / rms` (called from `graph/python.rs`, not a pipeline pass).
- `dot/` - the dot passes (`canonicalize`, `fold_reshapes`, `fuse`, `rows`, `split_k`, `strength`).
