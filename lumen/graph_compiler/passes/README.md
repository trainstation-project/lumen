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

## Frontend (every device, before staging)

1. `preserve_precision` - drop `cast(cast(x, b), a)` round trips for forward readers (flag
   `preserve_precision`).
2. `cse` - common subexpression elimination (never of custom calls or collectives).
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
12. `dot_strength` - vector-vector dots become multiply plus reduce-sum [mps, triton, tpu].
13. `merge_dots` (`MergeDotsLate`) - dot merging after simplification, packing whole vector lanes [tpu].
14. `neural_engine` - offload float16 GEMM regions over fixed weights to the Apple Neural Engine [mps].
15. `linear_reordering` - move a per-row or per-column factor of a matmul operand to its output [mps, triton].
16. `split_k` - split long-K, few-tile matmuls into partial products over K chunks [mps, triton].
17. `canonicalize_dots` - transpose dot operands into the layout the matmul kernels read [mps, tpu].
18. `normalization_diamonds` (`FusedDiamonds`) - replace normalization rows with `row` fusions [tpu].
19. `dot_rows` - fuse a matmul into the `row` fusion that reads its whole rows [tpu].
20. `dynamic_update_slices` (`Updates`) - chains of `dynamic_update_slice` become one in-place fusion [mps, triton, tpu].
21. `normalization_diamonds` (`Diamonds`) - find normalization rows and keep them for fusion [mps, triton].
22. `fuse_dots` - fuse TPU matmuls with their slices, stacked lhs and epilogues [tpu].
23. `place_concats` - let producers write directly into their concatenation slot [mps, triton, tpu].
24. `fusion` - group elementwise chains, reductions, rows and epilogues into `Fusion` kernels [cuda, mps, triton, tpu].
25. `wide_rows` - GEMMs feeding rows wider than a tile also write the row's partial reductions [triton].
26. `place_views` - slices read only by GEMMs become strided views [mps, triton].
27. `alone` - wrap leftover nodes that only a generated kernel can compute into one-node fusions [cuda, mps, triton].
28. `concats` - copy the concatenations fusion could not place in place [mps].
29. `horizontal_fusion` - merge independent small kernels of the same kind into one launch [triton, tpu].

## Other files

- `mod.rs` - the `Module`, `DevicePass` and `CompilerPass` traits.
- `identity.rs` - the identity pass.
- `elementwise.rs` - building elementwise fusion bodies from node groups (used by fusion).
- `multi_output.rs` - `Readers`: a fusion also writes a value other fusions read as an extra output.
- `rms_norm.rs` - `rms_norms` detects RMS norms written as primitives, so autograd can differentiate
  them from saved `1 / rms` (called from `graph/python.rs`, not a pipeline pass).
- `dot/` - the dot passes (`canonicalize`, `fold_reshapes`, `fuse`, `rows`, `split_k`, `strength`).
