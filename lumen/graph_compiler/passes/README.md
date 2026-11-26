# graph_compiler/passes

Every graph rewrite the compiler runs between tracing and codegen. Each device lists the passes it runs, in order, in
its own pass list (`graph_compiler/pass_container/<device>.rs`); the passes are shared.

## How a pass is written

- `Module` is what a pass rewrites: a graph, its `Options` and the device's `CompilerConfig`.
- `CompilerPass<M>` has `name` (used for profiler ranges and dump files) and `apply`. A device's list builds each
  pass it runs and pushes it, inside an `if` on the config flags it runs with; a pass reading its device's answers
  through a trait is built with `X::of::<Device>()`.
- A pass whose code is the same on every device that runs it is one file, `passes/<pass>.rs`, with no trait: the
  device lists that run it add it (`swap_pairs.rs`, `sink_reshapes.rs`). A value a device gives it is a field the
  list sets (`linear_reordering.rs`'s row tile, `updates.rs`'s chain lengths, `merge_dots`' `Target`).
- A pass with code of its own per device is a folder, `passes/<pass>/`: `mod.rs` holds the shared matching and
  rewriting and the trait the pass reads its device answers through (what is fusible, kernel names, limits); each
  device implements it in `passes/<pass>/<device>.rs` (`cpu`, `cuda`, `mps`, `triton`, `tpu`; `mps` only when MPS
  is linked). A device runs it where its list adds it, which needs that impl.
- The flags a pass runs with, and a device's own conditions (running a pass only with `fuse`), are in each list's
  `if`s, not the pass.

## Device passes, in order (devices whose lists add it in brackets)

1. `preserve_precision` - drop `cast(cast(x, b), a)` round trips for forward readers (flag `preserve_precision`);
   before CSE, which would share the casts forward and gradient readers keep apart [every device].
2. `cse` - common subexpression elimination (never of custom calls or collectives: each collective is a message
   every process must issue) [every device].
3. `simplify` (`matched: false`) - algebraic simplification without structural rewrites (identities, such as an
   update as large as its operand being the update) [every device].
4. `unsupported` - reject ops the device has no kernel for [cpu, cuda, mps, triton, tpu].
5. `take_along_axis` - lower `take_along_axis` to a flat `gather` [cuda, mps, triton].
6. `chunked_cross_entropy` - replay a cross entropy per row chunk when the logits are already chunked [mps, triton].
7. `merge_dots_backward` - merge gradient dots that share an operand [cuda, mps, triton, tpu].
8. `merge_dots` - merge forward dots sharing an operand into one dot against a packed weight block [cuda, mps, triton].
9. `flash_attention` - replace `softmax(QK^T)V` with a `flash_attention` fusion [cuda, mps, triton, tpu].
10. `flash_attention_backward` - replace attention gradients with backward fusions [mps, triton, tpu].
11. `attention_layouts` - fold attention output layout chains and `delta` into the fusions [tpu].
12. `linear_cross_entropy` - replace `cross_entropy(h @ w, target)` with one `LinearCrossEntropy` op [mps, triton].
13. `fold_dot_reshapes` - fold a reshape flattening leading dims into a 2-D dot's lhs [tpu].
14. `simplify` (`matched: true`) - full simplification, including structural rewrites [every device, CPU too].
15. `swap_pairs` - swaps of a last axis's partners written as slices and a concatenation become `SwapPairs { distance }`: adjacent pairs (`distance` 1, an interleaved rotary's), or each `d` elements' halves (`d / 2`, a rotate-half rotary's; a negated half becomes a sign multiply). Kernels read the partner at `j ^ distance`; a Triton GEMM accumulator swaps adjacent pairs in registers [cuda, mps, triton, tpu].
16. `sink_reshapes` - compute a matmul product's elementwise readers before the reshape splitting its last axis, so they stay in the GEMM's epilogue [tpu].
17. `dot_strength` - vector-vector dots become multiply plus reduce-sum [mps, triton, tpu].
18. `merge_dots` - TPU's dot merging, after simplification, packing whole vector lanes [tpu].
19. `neural_engine` - offload float16 GEMM regions over fixed weights to the Apple Neural Engine [mps].
20. `linear_reordering` - move a per-row or per-column factor of a matmul operand to its output [mps, triton].
21. `split_k` - split long-K, few-tile matmuls into partial products over K chunks [mps, triton].
22. `canonicalize_dots` - transpose dot operands into the layout the matmul kernels read [mps, tpu].
23. `normalization_diamonds` (`FusedDiamonds`) - replace normalization rows with `row` fusions [tpu].
24. `dot_rows` - fuse a matmul into the `row` fusion that reads its whole rows [tpu].
25. `gather_rows` - join a row gather (an embedding lookup) with the `row` fusion reading its rows, whose kernel
    fetches them itself; the gathered value an extra output when the residual reads it [tpu].
26. `dynamic_update_slices` (`Updates`) - chains of `dynamic_update_slice` become one in-place fusion [mps, triton, tpu].
27. `normalization_diamonds` (`Diamonds`) - find normalization rows and keep them for fusion [mps, triton].
28. `fuse_dots` - fuse TPU matmuls with their slices, stacked lhs and epilogues [tpu].
29. `place_concats` - let producers write directly into their concatenation slot [mps, triton, tpu].
30. `fusion` - group elementwise chains, reductions, rows and epilogues into `Fusion` kernels [cuda, mps, triton, tpu].
31. `wide_rows` - GEMMs feeding rows wider than a tile also write the row's partial reductions [triton].
32. `place_views` - slices read only by GEMMs become strided views [mps, triton, tpu].
33. `alone` - wrap leftover nodes that only a generated kernel can compute into one-node fusions [cuda, mps, triton, tpu].
34. `concats` - copy the concatenations fusion could not place in place [mps, tpu].
35. `horizontal_fusion` - merge independent small kernels of the same kind into one launch [triton, tpu].

## Other files

- `mod.rs` - the `Module` and `CompilerPass` traits.
- `elementwise.rs` - building elementwise fusion bodies from node groups (used by fusion).
- `multi_output.rs` - `Readers`: a fusion also writes a value other fusions read as an extra output.
- `gather_rows.rs` - the `gather_rows` pass (a single file: its code is the same wherever it runs).
- `rms_norm.rs` - `rms_norms` detects RMS norms written as primitives, so autograd can differentiate
  them from saved `1 / rms` (called from `graph/ir/python.rs`, not a pipeline pass).
- `dot/` - the dot passes (`canonicalize`, `fold_reshapes`, `fuse`, `rows`, `split_k`, `strength`, `wide_rows`) and
  each device's answers to them (`mps.rs`, `triton.rs`, `tpu.rs`).
- Folders with each device's answers beside the pass: `alone/`, `attention/`, `diamonds/`, `fusion/`,
  `horizontal/`, `place_concats/`, `place_views/`, `unsupported/` (each with a README); `apple_neural_engine/` and
  `merge_dots/` group their files.
