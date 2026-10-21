# `lumen/codegen/mps`

Metal kernels generated from fused graphs (and, in [`ane/`](ane/README.md),
Core ML programs for the Neural Engine). The MPS graph compiler
(`lumen/graph_compiler/mps`) groups primitives into `Primitive::Fusion`
nodes; its `codegen` pass (`graph_compiler/mps/kernel_codegen.rs`) calls
this module once per fusion body and gets back `(name, Metal source)`. The
source is compiled and launched by the MPS runtime (`lumen/ops/mps`); this
module only writes text.

Generated kernels are thin: most instantiate a hand-written template from
`lumen/ops/**/mps/*.metal` (the elementwise prelude, reductions, the
simdgroup matmul, flash attention) with a generated functor or input
struct that computes the fused primitives per element.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Kernel-kind dispatch (`kernel`), naming and formatting (`named`, `formatted`), kernel parameters (`io_params`), output views (`writes_views`, `written`), Metal types and functors (`metal_type`, `functor`, `constant`); re-exports each kind's predicates. |
| `emitter.rs` | The `Emitter`: each fused value as a Metal expression at an index (`value`), hoisting values the same across a row; `gather_index`. |
| `reduction.rs` | Reductions with prologue and epilogue (`reduction`, `reduction_root`, `split_launches`, `has_epilogue`). |
| `transpose.rs` | Tiled transposes (`TransposeTiling`, `transpose_tiling`, `transpose_kernel`). |
| `row.rs` | Row kernels for normalization diamonds (`row_kernel`, `row_reductions`, `row_partials`, `simd_rows`, `row_groups`) and the helpers they share with GEMM rows (`softmax_pair`, `kept_values`, `depends_on`). |
| `gemm_row.rs` | A dot whose rows a row kernel normalizes (`row_dot`, `gemm_row_kernel`, `gemm_rows`, `gemm_tiles`, `tile_scratch`). |
| `gemm.rs` | GEMM with epilogues (`gemm_dot`, `gemm_kernel`), column regions (`gemm_regions`, `Columns`), paired, interleaved and expanded outputs. |
| `attention.rs` | Flash attention forward and backward (`attention_kernel`, `attention_backward_kernel`), dropout and score functors, block choice. |

## Subfolders

[`ane/`](ane/README.md) — lowering a fusion body to a Core ML ML Program
(MIL protobuf) for the Apple Neural Engine.

## Entry points

* `kernel(body, by_value, config, views)` — any loop/reduction/row/GEMM
  fusion. Picks the kernel kind (first match wins):
  1. `row_dot(body)` → `gemm_row_kernel` (a dot whose rows a row kernel normalizes, in its epilogue)
  2. `gemm_dot(body)` → `gemm_kernel` (a contraction with its epilogue)
  3. non-empty `row_reductions(body)` → `row_kernel` (normalization diamonds)
  4. `reduction_root(body)` → `reduction` (a reduction with prologue/epilogue)
  5. `transpose_tiling(body)` → `transpose_kernel` (tiled transpose)
  6. otherwise the elementwise loop kernel.
* `attention_kernel(body, &Attention)` — forward flash attention.
* `attention_backward_kernel(body, &Backward)` — attention backward.

`by_value[k]` marks body inputs passed as constants (`setBytes`: runtime
scalars, a dropout's seed/offset) rather than device buffers. `views` maps
an output position to `(offset, strides)` in a larger buffer (see *Output
views*).

## Kernel kinds

**Loop (elementwise).** One thread per output element, flat index `j`,
using the `ELEMENTWISE_ARGS` / `FOR_EACH_ELEMENT` macros from
`ops/mps/kernels.metal`. Multi-output fusions write `out1`, `out2`, ... at
the same index. Layout primitives (reshape, broadcast_in_dim, transpose,
slice), gather, concatenate, iota, full and random_bits are computed by
index arithmetic inside the loop, never materialized.

**Transpose-tiled** (`TransposeTiling`, XLA's transpose emitter). When a
loop fusion has a transpose ("hero") whose operand's innermost dimension
is not the output's and both span at least `MIN_TILED` (16) elements, a
threadgroup loads a 32x33 (`TILE` + 1 padding) tile of each hero into
threadgroup memory reading contiguously, barriers, then writes the output
contiguously. 16x16 threads, 2x2 elements each; grid from
`TransposeTiling::grid()`. At most `MAX_HEROES` (4) transposes per tile
(threadgroup-memory bound; `horizontal.rs` respects it).

**Row kernels** (`row_kernel`; normalizations, `graph_compiler/mps/diamonds.rs`).
A fusion whose non-root reductions run over the last dimension. One
threadgroup per row, or, for rows of at most `SIMD_ROW` (256) elements
with float/half/int accumulators, one SIMD group per row and `SIMD_ROWS`
(8) rows per threadgroup (`simd_rows`: `simd_sum`/`simd_max`, no barriers).
One pass per reduction (independent reductions share a pass), then a final
pass writing the output. Details:
* Values the same across a row (reductions' results, constants,
  `sqrt(mean + eps)`) are *invariant*: computed once per row into
  `hoisted`, held in `row_locals`.
* With `config.online_softmax`, a softmax's max and sum (`softmax_pair`)
  run as one online pass.
* When each thread holds at most `config.row_cache` elements, values an
  earlier pass computed are kept in registers (`kept_values`) so the row
  is read once.
* `row_partials`: reductions over *other* axes (column sums over blocks of
  rows, `diamonds::partials`); then `row_groups` is the partials' leading
  dimension.

**Reductions** (`reduction`, XLA's reduce input fusion). The fusion's root
is a `reduce_sum`/`reduce_max` reached through elementwise ops/reshapes.
Generates a `NAME_input` struct whose `operator[]` computes each input
element from the fusion's inputs, and instantiates `reduce_rows` /
`reduce_cols` / `reduce_grouped` from `ops/reduce/mps/kernels.metal` per
`reduce::layout`. An epilogue (cast, a mean's division) becomes a
`NAME_epilogue` functor. When `reduce::splits` says the reduction splits,
the epilogue is applied in a second kernel `NAME_final` over the partials
(`split_launches` names the two launches for profiling).

**GEMM with epilogue** (`gemm_kernel`). Instantiates `matmul_sg_impl` from
`ops/dot_general/mps/kernels.metal` three times — `NAME` (128x64 tiles),
`NAME_mid` (64x64), `NAME_small` (32x32) — and the encoder picks one. The
epilogue (bias, activation, residual, cast) is a `NAME_epi` functor of the
dot's value `r` and the flat index. Variants:
* *paired* (`gemm_pair`): the dot's only readers are slices taking its two
  halves along the last dimension (a gated MLP's merged `w1|w3`); the
  functor takes `r1, r2`.
* *expanded* (`gemm_expanded`): the output concatenates two values of the
  dot's shape along the last dimension (the gated pair's backward); the
  functor returns a `Pair<T>`.
* Other outputs of the fusion are stored inside the functor at index `i`.

**GEMM row kernel** (`gemm_row_kernel`, fusing a row kernel into the GEMM
epilogue). `row_dot` accepts a single unbatched float dot accumulating in
float32 whose `[rows, n]` value feeds a SIMD row kernel without partials,
with `n <= ROW_TILE` (64) and both operands fusion inputs. The body is
rebuilt with the dot's value as an extra input, a row kernel is generated
for it, and `gemm_rows` runs the matmul on 64x64 tiles, rounds and stages
each tile in threadgroup memory, and runs the row code from there
(`read_staged`): the dot's value never reaches device memory.

**Attention.** `attention_kernel` instantiates `flash_attention`, or
`attention_decode` when `decodes(a)` (at most 8 queries), from
`ops/attention/mps/forward.metal`, with per-operand stride indexers
(`ix_method`), a score functor (the traced roundings/casts/scale,
`score_functor`) and an optional dropout functor (`dropout_functor`; the
random state is the body's last two inputs, `random_state`). Key block:
largest of 32/16/8 fitting 28 KB (`key_block`).
`attention_backward_kernel` instantiates `flash_attention_dkdv` (dV and dK,
plus dQ when `backward_block_with_dq` finds a block that fits) or
`flash_attention_dq` from `ops/attention/mps/backward.metal`.

## The `Emitter`

Builds the per-element body shared by every kind. `value(v, idx)` returns a
Metal local holding value `v` at row-major index `idx`, emitting
statements into `lines` and memoizing by `(var, index)`; index
expressions are memoized too (`index`). `operand_index` maps an output
index through a layout primitive (XLA's indexing maps), using
`gather_index`, which merges contiguous dimensions to save div/mod.
Fields `invariant` / `row_locals` / `hoisted` let callers pin a value to a
name computed once (a row's reduction, a GEMM's `r`, a transpose tile
slot). Elementwise ops map to functors in `ops/mps/kernels.metal`
(`functor`); constants are emitted from the exact bits `full` would fill
(`constant`). Anything else hits `unreachable!("... is not loop-fusible")`.

## Naming: `named`

Every generator writes `NAME` where the kernel (and its helper structs)
are named. `named` first re-indents the source by brace depth
(`formatted`), hashes it with `DefaultHasher`, and substitutes
`fusion_<16 hex>`. Identical bodies therefore produce identical names and
are compiled once (the `codegen` pass shares one kernel among fusions with
the same source). Second launches are suffixed (`NAME_final`, `NAME_mid`,
`NAME_small`).

## Output views

`writes_views(body)` is true for loop and transpose-tiled kernels (each
element written at its own flat index `j`), not for GEMM, row or reduction
kernels. For those, `views` lets an output be written at an offset and
strides inside a larger buffer — an operand of an in-place `concatenate`
(see `Graph::placements` in [`lumen/graph`](../../graph/README.md)).
`written` turns `j` into that strided address.

## Buffer ABI (`io_params`)

Buffers in order: non-by-value inputs `in{k}`, `out`, extra outputs
`out1..`, then by-value inputs as `constant T &in{k}`; kernel-specific
arguments (reduction counts, matmul params `p`) start at the returned
index. The encoders in `graph_compiler/mps` and `ops/*/mps` bind
arguments in this exact order — change both together.

## Invariants & gotchas

* Exact dtypes: accumulators are the graph's `accum_dtype`; nothing is
  widened or narrowed beyond what the body says. Online softmax and flash
  attention are the only reorderings, gated by `CompilerConfig` flags.
* The kind predicates are also used by other passes (`horizontal.rs`,
  `place_views.rs`, `kernel_codegen.rs`); changing what e.g. `gemm_dot` or
  `transpose_tiling` accepts changes fusion decisions, not just codegen.
* Kernel names depend on the formatted source text only; any change to a
  template string renames kernels (harmless, but visible in profiles).
* Generated kernels read gather indices as given, no clamping; with
  `lumen.config.compiler.safe_kernels` the graph compiler clamps them
  first (`graph_compiler/mps/bounds.rs`).
* `row_kernel`'s doc mentions `MAX_CACHED`; the actual limit is
  `config.row_cache`.

## Related

* [`ane/`](ane/README.md) — the Neural Engine lowering.
* [`lumen/graph`](../../graph/README.md) — `Graph`, `Primitive`, `placements()`.
* [`lumen/graph_compiler/mps`](../../graph_compiler/mps/README.md) — fusion passes and `kernel_codegen.rs` (the caller); `graph_compiler/attention.rs` defines `Attention` / `Backward`; `graph_compiler/config.rs` the flags read here.
* [`lumen/ops`](../../ops/README.md) — the Metal templates and launch code (`ops/mps`, `ops/reduce/mps`, `ops/dot_general/mps`, `ops/attention/mps`).
