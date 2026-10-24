# `lumen/graph_compiler/mps`

The MPS (Apple Metal) graph compiler. It rewrites a graph that already went
through the frontend (`cse`, `simplify`) into one whose steps are either
lumen's primitive kernels (`lumen/kernels/*/mps`) or `Primitive::Fusion`s whose
kernels are generated as Metal source by `lumen/codegen/mps`, compiles all
generated sources into one Metal library, and plans the result with
`Plan::compile_with` (with kernel scratch in the workspace, so no kernel
allocates). At run time, `lumen/kernels/mps` calls `encode` here for every
fusion step.

The whole module only exists with `cfg(lumen_mps_linked)` (macOS, shim
built). The structure follows XLA's GPU compiler: an `HloPassPipeline` of
rewrites, most named after their XLA counterpart in their `//!` doc.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `MpsModule`, `Kernel`, `PENDING`/`request`, `pipeline()`, `compile`, `compile_kernels` (+ `PRELUDE`, `SOURCES`, `fusion_source`), `encode` (a step with a strided input runs `strided_variant`: its kernel with that input's load prologue through its strides, compiled once a layout, `VARIANTS`), `fusion_scratch_bytes`; `copy` and `fill`, an MPS tensor's `copy_` / `contiguous` and `fill_` as one loop on the GPU (a load prologue and a store epilogue in the bytes' unsigned type, so any dtype). |
| `flash_attention_backward.rs` | Pass `flash_attention_backward`: attention backward as two fusions, or one with dQ added atomically unless `deterministic`. |
| `split_k.rs` | Pass `split_k`: splits the K of small-M/N, long-K float dots into chunks (XLA `SplitKRewriter`); `atomic_dot` for the atomic variant. |
| `canonicalize_dots.rs` | Pass `canonicalize_dots`: transposes dot operands into matmul form as a separate step. |
| `diamonds.rs` | Pass `normalization_diamonds` (`Diamonds`): the softmax/RMS/layer-norm style row diamonds (`graph_compiler/common/diamonds.rs`, over MPS's `fusible`) and their `partials` (column sums a row's kernel adds up too); stores `Row`s in `MpsModule::rows`. |
| `place_concats.rs` | Pass `place_concats`: marks concatenates whose operands' kernels can write them in place; `in_place_concats` also re-checked by `Codegen`. |
| `fusion/mod.rs` | Pass `loop_fusion` (`LoopFusion`): `fuse` runs the phases below in order, then horizontal fusion; also `fusible`, `label`. |
| `fusion/analysis.rs` | What every phase reads: each value's producer and readers, liveness, outputs, what is fusible (`Analysis`); `depends`, `recomputed`. |
| `fusion/rows.rs` | Normalization rows grown by the elementwise nodes reading their roots (`extended`). |
| `fusion/reduction_epilogues.rs` | A reduction's elementwise consumers, its epilogue (`reduction_epilogues`). |
| `fusion/dot_epilogues.rs` | A dot's epilogue: its column regions, or the chain of elementwise nodes after it, and the values it writes besides (`dot_epilogues`). |
| `fusion/hosts.rs` | Which nodes start a fusion (`roots`) and which values a fusion writes besides its root (`hosts`, multi-output fusion). |
| `fusion/emit.rs` | Each fusion's members and reads within a kernel's bindings (`fusions`), rows joined to the dot they read (`row_dots`), and the fused graph (`emitted`). |
| `horizontal.rs` | MPS's horizontal fusion `Target` (`../common/horizontal.rs`): which fusions join (no reductions, attention, atomic split-K or GEMM; at most 1M elements), keyed by element count (and shape for tiled transposes), within Metal's 30 buffers and the transpose tiles' heroes. |
| `alone.rs` | Pass `alone`: every elementwise node, gather and concatenate left outside a fusion becomes a one-node fusion, run by a generated loop kernel (none has a primitive kernel on MPS); `wrapped` (also used by `Codegen`). |
| `kernel_codegen.rs` | Pass `codegen` (`Codegen`): generates Metal source for every `pending#k` fusion and renames it to its kernel. |
| `autotune.rs` | MPS's two kernel families and its timer for the shared `Autotuner` (`../common/autotune.rs`). `Loops`: steps whose generated kernel has candidates (`codegen::tunings_of`) and 64K to 16M output elements, each candidate generated as Metal; its launch runs the fusion's `encode_with` with the candidate's kernel. `Gemms`: each float matmul with 64K or more outputs and no strided operand (a `dot_general` step, or a fusion launched as a GEMM), a kernel per `GemmConfig` (`kernels/dot_general/mps/config.rs`: tile, K step, threads, SIMD-group columns) from `dot_general::mps::matmul_source` or the fusion's own `codegen::gemm_variant`, plus, for a `dot_general` accumulating in float32, the `matmul_mpp` candidates where Metal Performance Primitives compile (`mpp_available`, probed once; compiled apart, as their header changes other kernels' math) and the `matmul_skinny` ones for 32 rows or fewer (tuned at any size); its launch is `dot_general::mps::encode_tuned`, or the fusion's `encode_with` given the kernel and config, with the config's threads (`Grid::GroupsOf`). `Mps`, the timer: `encode` under `capture` over zeroed cold `Buffers`, each launch timed on the GPU. |
| `place_views.rs` | Pass `place_views`: marks slices that dots read in place as strided views; `fusion_reads_strided`; `loads_strided` (an input only the load prologues read, not a dot's or attention's operand: any strides). |
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
`MpsModule`; the device-independent ones from [`../common/`](../common/README.md),
of any module). The order is load-bearing; the comments in
`pipeline()` say why each sits where it does.

| # | Pass name | Runs when | File | What it does |
| --- | --- | --- | --- | --- |
| 0 | `take_along_axis` | always | `../common/take_along_axis.rs` | Gathers along an axis (`F.gather`) as gathers of the operand's flat positions, which the MPS's kernels (and the cross-entropy patterns) read. |
| 1 | `chunked_cross_entropy` | `fuse && fused_linear_cross_entropy` | `../common/chunked_cross_entropy.rs` | Chunked CE's rows from each chunk's logits, before dots merge or the forward is matched as a linear CE. |
| 2 | `merge_dots_backward` | `fuse && merge_dots && contraction_epilogues` | `../common/merge_dots_backward.rs` | Merge a gated pair's/group's backward dots (before `merge_dots`, which reads its weights' concatenate as the forward's block). |
| 3 | `merge_dots` | `fuse && merge_dots` | `../common/merge_dots.rs` | Merge dots over packable parameters into blocks (`Target`: `fusion::fusible` readers, views, 8 columns); fills `MpsModule::packed`. |
| 4 | `flash_attention` | `fuse && flash_attention` | `../common/flash_attention.rs` (`Attentions` in `mod.rs`) | Attention → one flash-attention fusion (before dots are put in matmul form). |
| 5 | `flash_attention_backward` | `fuse && flash_attention` | `flash_attention_backward.rs` | Attention backward → flash-attention backward fusion(s). |
| 6 | `linear_cross_entropy` | `fuse && fused_linear_cross_entropy` | `../common/linear_cross_entropy.rs` | CE of unshared logits → `LinearCrossEntropy`. |
| 7 | `clamp_indices` | `safe_kernels` | `../common/clamp_indices.rs` | Clamp indices (after linear CE matching, which reads the gather's index). |
| 8 | `simplify` | always | `../common/simplify.rs` | `Simplify { matched: true }`: now that patterns are matched, swap transposes of dots, move reshapes/transposes/casts. |
| 9 | `neural_engine` | `neural_engine` | `ane/mod.rs` | Offload float16 regions on fixed weights to Core ML (whole dots, before split-K). |
| 10 | `dot_strength` | always | `../common/dot_strength.rs` | Vector dots → product + sum. |
| 11 | `split_k` | `split_k` | `split_k.rs` | Split long-K dots; atomic unless `deterministic`. |
| 12 | `canonicalize_dots` | always | `canonicalize_dots.rs` | Dot operands into matmul form. |
| 13 | `normalization_diamonds` | `fuse && normalization_diamonds` | `diamonds.rs` | Row kernels' `Row`s for loop fusion. |
| 14 | `place_concats` | `fuse` | `place_concats.rs` | Mark in-place concatenates (loop fusion leaves them alone). |
| 15 | `loop_fusion` | `fuse` | `fusion/` (+ `horizontal.rs` if `horizontal_fusion`) | Fuse; clears `rows`; may drop runtime scalars. |
| 15′ | `alone` | always | `alone.rs` | Each elementwise node, gather or concatenate left alone its own fusion; keeps a runtime scalar by value only if fusions alone read it. |
| 16 | `codegen` | always | `kernel_codegen.rs` | Generate Metal for each pending fusion; unmark concatenates that can no longer be in place. |
| 17 | `place_views` | always | `place_views.rs` | Mark slices dots read in place as views. |

After the plan is built, `autotune` (flag `autotune`, on by default) picks
each generated kernel's launch: see *Autotuning*.

## Autotuning

A generated kernel's launch (`codegen::Tuning`: a loop's elements per
thread, strided or contiguous, and threadgroup size) is written into its source as constants, so
dumped code shows what runs, and `encode` reads it from the kernel's entry
in `LAUNCHES` (recorded with its launch kind). Codegen writes a default
(`codegen::tuning_of`). With `autotune` on, the shared `Autotuner`
(`../common/autotune.rs`) tunes the plan's `Loops` and then its `Gemms`
(`autotune.rs`) when it compiles: for each step whose kernel has candidates (`codegen::tunings_of`)
and 64K to 16M output elements, each candidate is generated and all are
compiled together, `encode` runs once per candidate under
`kernels::mps::capture`, which records its launches instead of submitting
them, and each recorded launch is timed on the GPU alone
(`lumen_mps_time_pipeline`: one command buffer, the dispatch repeated
about 1 ms' worth with a barrier before each, as the stream runs it, GPU
start to end, best of two). The buffers are zeroed (zero indices are in
range), outside the static allocator (`allocator::Scratch`, fewer copies or
no tuning when Metal is short of memory), with enough copies of the inputs,
about 64 MB and at least two, that the repeats read them cold, as a model's kernels do once
their inputs are out of cache: timed hot, an 8 MB input stays in the GPU's
cache and the candidates rank differently. Encode launches the fastest
candidate's launch, set on the plan's step (`kernels::mps::encode` runs it, `Step::launch`); steps of the
same kernel and types are timed once. Loops and GEMMs have candidates so far.

`neg` per call on an M4 Pro, back to back on cold inputs, best of 3,
`autotune` off → on: float32 64K 3.11 → 2.73 µs, 256K 7.53 → 7.01 µs,
1M 26.8 → 24.5 µs; bfloat16 64K 2.82 → 2.05 µs, 256K 5.56 → 4.25 µs, 1M
15.8 → 13.1 µs; from 16M every launch streams at DRAM's bandwidth.
bfloat16 wants few threadgroups (4M elements: 39 µs in groups of 1024,
100 µs in groups of 256, a thread 4 elements apart either way). Tuning
adds 20 to 120 ms to a compile, once per kernel and length.

Other flags are read inside passes rather than gating them:
`contraction_epilogues` (flash attention's and GEMM epilogues in
`fusion/`), `reduction_epilogues` and `multi_output_fusion` (`fusion/`),
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
  strides (`kernels::dot_general::mps::reads_strided`), and that are not
  outputs or slices of another view. The step reads them via `Step::views`.

## What stays in `mod.rs`

- **`compile`**: build `MpsModule`, run `pipeline(config)`,
  `compile_kernels` (prelude + all sources into one library through
  `lumen_mps_compile_kernels` in `lumen/kernels/mps/shim.mm`; sources are kept
  in `SOURCES` for `fusion_source`, which the graph viewer shows), extend
  `parameters` to cover packed blocks, plan with
  `scratch: Some(kernels::mps::scratch_bytes)`, then relabel dots reading a
  packed block (or a value computed elementwise from one) as
  `Nx dot_general`.
- **`encode(step, …)`**: launches a fusion step. Dispatch order:
  `coreml_*` → `ane::encode`; attention; attention backward (zeroing dQ
  first when it is added atomically); atomic split-K dot (zero output, then
  `matmul_atomic_<dtype>`); GEMM with epilogue (`_small` / `_mid` tile
  variants, or a row-dot grid); row kernel; reduction fusion
  (`kernels::reduce::mps::encode_reduction`, `_final` launch for epilogues);
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
- A Metal kernel binds at most 31 buffers; `fusion/` caps a fusion's
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
- [`../../kernels/`](../../kernels/) — primitive MPS kernels, `launch`, `scratch_bytes`, matmul planning.
- [`../../graph/`](../../graph/) — `Primitive` flags (`in_place`, `view`), `placements`, `slot`.
