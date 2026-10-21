# `lumen/graph_compiler/mps`

The MPS (Apple Metal) graph compiler. It rewrites a graph that already went
through the frontend (`cse`, `simplify`) into one whose steps are either
lumen's primitive kernels (`lumen/ops/*/mps`) or `Primitive::Fusion`s whose
kernels are generated as Metal source by `lumen/codegen/mps`, compiles all
generated sources into one Metal library, and plans the result with
`Plan::compile_with` (with kernel scratch in the workspace, so no kernel
allocates). At run time, `lumen/ops/mps` calls `encode` here for every
fusion step.

The whole module only exists with `cfg(lumen_mps_linked)` (macOS, shim
built). The structure follows XLA's GPU compiler: an `HloPassPipeline` of
rewrites, most named after their XLA counterpart in their `//!` doc.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `MpsModule`, `Kernel`, `PENDING`/`request`, `pipeline()`, `compile`, `compile_kernels` (+ `PRELUDE`, `SOURCES`, `fusion_source`), `encode`, `fusion_scratch_bytes`. |
| `flash_attention.rs` | Pass `flash_attention`: each matched attention one fusion (`attention::fuse`), `Kernel::Attention`. |
| `flash_attention_backward.rs` | Pass `flash_attention_backward`: attention backward as two fusions, or one with dQ added atomically unless `deterministic`. |
| `split_k.rs` | Pass `split_k`: splits the K of small-M/N, long-K float dots into chunks (XLA `SplitKRewriter`); `atomic_dot` for the atomic variant. |
| `canonicalize_dots.rs` | Pass `canonicalize_dots`: transposes dot operands into matmul form as a separate step. |
| `diamonds.rs` | Pass `normalization_diamonds` (`Diamonds`): finds softmax/RMS/layer-norm style row diamonds and their `partials` (XLA `SoftmaxRewriterTriton`); stores `Row`s in `MpsModule::rows`. |
| `place_concats.rs` | Pass `place_concats`: marks concatenates whose operands' kernels can write them in place; `in_place_concats` also re-checked by `Codegen`. |
| `fusion.rs` | Pass `loop_fusion` (`LoopFusion`): loop/reduction/epilogue/multi-output fusion (`fuse`), then horizontal fusion; also `fusible`, `label`. |
| `horizontal.rs` | Horizontal loop fusion: independent same-size small loop fusions run as one multi-output kernel. |
| `alone.rs` | Pass `alone`: every elementwise node and concatenate left outside a fusion becomes a one-node fusion, run by a generated loop kernel (neither has a primitive kernel on MPS); `wrapped` (also used by `Codegen`). |
| `kernel_codegen.rs` | Pass `codegen` (`Codegen`): generates Metal source for every `pending#k` fusion and renames it to its kernel. |
| `place_views.rs` | Pass `place_views`: marks slices that dots read in place as strided views; `fusion_reads_strided`. |
| `tests.rs` | End-to-end MPS compiler tests against the reference executor (skip without Metal via `plan_tests::mps::available`). |

## Subfolders

- [`ane/`](ane/README.md) — Neural Engine offload through Core ML (`neural_engine` flag).

## The pipeline, in order

Before them, `compile` refuses what a graph may have but the MPS has no
kernel for (`unsupported`): exp, log, sqrt, tanh and logistic of bfloat16
(Metal's take and return float, and a program runs as traced, so it
converts to float32 itself), and dots of two dtypes. The rules are the
MPS's alone: the CPU's reference and a TPU (whose v6e computes those in
bfloat16, and multiplies bfloat16 by float8) run them.

`pipeline(config)` builds these passes, each a `CompilerPass` (MPS ones of
`MpsModule`; the device-independent ones from [`../passes/`](../passes/README.md),
of any module). The order is load-bearing; the comments in
`pipeline()` say why each sits where it does.

| # | Pass name | Runs when | File | What it does |
| --- | --- | --- | --- | --- |
| 0 | `take_along_axis` | always | `../passes/take_along_axis.rs` | Gathers along an axis (`F.gather`) as gathers of the operand's flat positions, which the MPS's kernels (and the cross-entropy patterns) read. |
| 1 | `chunked_cross_entropy` | `fuse && fused_linear_cross_entropy` | `../passes/chunked_cross_entropy.rs` | Chunked CE's rows from each chunk's logits, before dots merge or the forward is matched as a linear CE. |
| 2 | `merge_dots_backward` | `fuse && merge_dots && contraction_epilogues` | `../passes/merge_dots_backward.rs` | Merge a gated pair's/group's backward dots (before `merge_dots`, which reads its weights' concatenate as the forward's block). |
| 3 | `merge_dots` | `fuse && merge_dots` | `../passes/merge_dots.rs` | Merge dots over packable parameters into blocks (`Target`: `fusion::fusible` readers, views, 8 columns); fills `MpsModule::packed`. |
| 4 | `flash_attention` | `fuse && flash_attention` | `flash_attention.rs` | Attention → one flash-attention fusion (before dots are put in matmul form). |
| 5 | `flash_attention_backward` | `fuse && flash_attention` | `flash_attention_backward.rs` | Attention backward → flash-attention backward fusion(s). |
| 6 | `linear_cross_entropy` | `fuse && fused_linear_cross_entropy` | `../passes/linear_cross_entropy.rs` | CE of unshared logits → `LinearCrossEntropy`. |
| 7 | `clamp_indices` | `safe_kernels` | `../passes/clamp_indices.rs` | Clamp indices (after linear CE matching, which reads the gather's index). |
| 8 | `simplify` | always | `../passes/simplify.rs` | `Simplify { matched: true }`: now that patterns are matched, swap transposes of dots, move reshapes/transposes/casts. |
| 9 | `neural_engine` | `neural_engine` | `ane/mod.rs` | Offload float16 regions on fixed weights to Core ML (whole dots, before split-K). |
| 10 | `dot_strength` | always | `../passes/dot_strength.rs` | Vector dots → product + sum. |
| 11 | `split_k` | `split_k` | `split_k.rs` | Split long-K dots; atomic unless `deterministic`. |
| 12 | `canonicalize_dots` | always | `canonicalize_dots.rs` | Dot operands into matmul form. |
| 13 | `normalization_diamonds` | `fuse && normalization_diamonds` | `diamonds.rs` | Row kernels' `Row`s for loop fusion. |
| 14 | `place_concats` | `fuse` | `place_concats.rs` | Mark in-place concatenates (loop fusion leaves them alone). |
| 15 | `loop_fusion` | `fuse` | `fusion.rs` (+ `horizontal.rs` if `horizontal_fusion`) | Fuse; clears `rows`; may drop runtime scalars. |
| 15′ | `alone` | always | `alone.rs` | Each elementwise node or concatenate left alone its own fusion; keeps a runtime scalar by value only if fusions alone read it. |
| 16 | `codegen` | always | `kernel_codegen.rs` | Generate Metal for each pending fusion; unmark concatenates that can no longer be in place. |
| 17 | `place_views` | always | `place_views.rs` | Mark slices dots read in place as views. |

Other flags are read inside passes rather than gating them:
`contraction_epilogues` (flash attention's and GEMM epilogues in
`fusion.rs`), `reduction_epilogues` and `multi_output_fusion` (`fusion.rs`),
`online_softmax` and `row_cache` (`lumen/codegen/mps`), `deterministic`
(split-K, attention backward).

## `MpsModule`

What the passes share besides the graph:

- `packed: Vec<(Vec<usize>, usize)>` — parameter blocks added by
  `merge_dots`, appended after the graph's inputs (members, dimension).
  Becomes `Plan::packed` and `PlanOptions::blocks`.
- `rows: Vec<diamonds::Row>` — diamond chains by node; only valid until the
  graph next changes (set by `normalization_diamonds`, consumed and cleared
  by `loop_fusion`).
- `scalars: Vec<usize>` — input positions of runtime scalars that fusion
  kernels may take by value (`setBytes`). Passes keep inputs in order so
  positions stay valid; `loop_fusion` re-fuses while it drops scalars read
  by a non-fusion step.
- `kernels: Vec<Kernel>` and `sources: BTreeMap<String, String>` — kernel
  requests and their generated Metal source.

## Kernel naming: `request` → `pending#k` → codegen

Fusion-creating passes don't generate code. They call
`request(&mut kernels, Kernel::…)`, which pushes the request and returns the
placeholder name `"pending#<index>"` (`PENDING`) used as the fusion's
`name`. `Kernel` is `Loop(by_value)` (which inputs are `setBytes` scalars),
`Attention` or `AttentionBackward`. The `codegen` pass later replaces each
placeholder with the real kernel name from `codegen::kernel`,
`attention_kernel` or `attention_backward_kernel` and records the source in
`sources`; fusions with identical kernels share one name and one source.
Until `codegen` has run, a fusion's name is only an index into `kernels`
(`place_concats` uses this to know each fusion's kind).

## In-place placement

Two primitive flags place values inside another value's memory instead of
copying; `Graph::placements()` turns them into `(value, View)` pairs, which
`Codegen`, `encode` and `plan.rs` all read:

- **`Primitive::Concatenate { in_place }`** — set by `place_concats` when
  every operand is read only by the concatenate and its producer (an
  attention backward kernel through its layout chain, or a loop fusion that
  writes views) can write it at its slot (`Graph::slot`) and at least one
  is an attention backward's. It runs *before* fusion, guessing which
  operands will become fusions; `codegen` re-runs `in_place_concats(..,
  false)` on the fused graph and unmarks any that no longer qualify,
  turning them into one-node fusions (`alone::wrapped`). Marked concatenates have
  no step: their operands' kernels get `output_views`.
- **`Primitive::Slice { view }`** — set by `place_views` (last pass) for
  slices read only by dots, or GEMM-epilogue fusions
  (`fusion_reads_strided`), that the matmul kernels can read at the slice's
  strides (`ops::dot_general::mps::reads_strided`), and that are not
  outputs or slices of another view. The step reads them via `Step::views`.

## What stays in `mod.rs`

- **`compile`**: build `MpsModule`, run `pipeline(config)`,
  `compile_kernels` (prelude + all sources into one library through
  `lumen_mps_compile_kernels` in `lumen/ops/mps/shim.mm`; sources are kept
  in `SOURCES` for `fusion_source`, which the graph viewer shows), extend
  `parameters` to cover packed blocks, plan with
  `scratch: Some(ops::mps::scratch_bytes)`, then relabel dots reading a
  packed block (or a value computed elementwise from one) as
  `Nx dot_general`.
- **`encode(step, …)`**: launches a fusion step. Dispatch order:
  `coreml_*` → `ane::encode`; attention; attention backward (zeroing dQ
  first when it is added atomically); atomic split-K dot (zero output, then
  `matmul_atomic_<dtype>`); GEMM with epilogue (`_small` / `_mid` tile
  variants, or a row-dot grid); row kernel; reduction fusion
  (`ops::reduce::mps::encode_reduction`, `_final` launch for epilogues);
  tiled transpose; plain elementwise grid. Inputs that are
  `Buffer::Scalar` are passed as bytes before the kernel's own arguments.
- **`fusion_scratch_bytes`**: workspace a reduction fusion's split
  reduction needs (0 for atomic split-K dots).

## Invariants & gotchas

- Pattern matchers (attention, linear CE, merge_dots_backward, merge_dots) must
  run before `simplify` and `canonicalize_dots` move transposes/casts, or
  they stop matching. Respect the order comments in `pipeline()`.
- Passes must not reorder or remove graph inputs: `scalars` and `packed`
  are input positions.
- Every pass after `place_concats` must preserve `in_place` concatenates
  (loop fusion leaves them alone).
- `rows` are node indices: any graph change between `normalization_diamonds`
  and `loop_fusion` invalidates them.
- `merge_dots` requires fusion: merged dots' readers read their slices in
  place.
- A Metal kernel binds at most 31 buffers; `fusion.rs` caps a fusion's
  inputs (`MAX_INPUTS`).
- Kernels read indices as given; clamping exists only behind
  `safe_kernels`.
- Flags that change rounding (`online_softmax`, `flash_attention`,
  `fused_linear_cross_entropy`, `split_k`, `neural_engine`, non-
  `deterministic` atomics) must stay behind their flags.

## Related

- [`..`](../README.md) — frontend, `Pass`/`Pipeline`, `Plan`, `attention.rs`.
- [`ane/`](ane/README.md) — Core ML offload used by pass 9 and `encode`.
- [`../../codegen/`](../../codegen/) — Metal source for fusions (`kernel`, `gemm_dot`, `row_reductions`, …).
- [`../../ops/`](../../ops/) — primitive MPS kernels, `launch`, `scratch_bytes`, matmul planning.
- [`../../graph/`](../../graph/) — `Primitive` flags (`in_place`, `view`), `placements`, `slot`.
