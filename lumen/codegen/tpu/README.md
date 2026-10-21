# `lumen/codegen/tpu`

What TPU kernels generate from a fusion's body, as `codegen/mps` is to
Metal: each kind's spec (the body's nodes, its operands' views and blocks)
for a hand-written template in `ops/<op>/tpu/` (a Pallas program, its
schedule), and the replay of those nodes while the template is traced
(`__init__.py`, the `Emitter` of a TPU: tracing inlines them, as Metal's
compiler inlines a functor). Templates: loops (`ops/elementwise/tpu`), rows
(`ops/row/tpu`), reductions (`ops/reduce/tpu`), the dot
(`ops/dot_general/tpu`) and attention (`ops/attention/tpu`).

Fusions come from the TPU graph compiler: loops and reductions from
`loop_fusion` (`graph_compiler/tpu/fusion/`), rows from `diamonds`
(normalization diamonds, XLA's `SoftmaxRewriterTriton`), the dot's
epilogues from `fuse_dots`, attention's from `flash_attention`. Loops take
layout primitives too (`indexing.rs`): a broadcast, a reshape of unit
dimensions, a slice at whole blocks, a transpose of major dimensions and
iota become where each operand is read. Gathers, update slices and reshapes
that move elements between tiles are not yet generated.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | What the kinds share: `fusible` (the primitives a loop fusion holds), `node` (a primitive as the spec's JSON) and `window` (an output's column window, for in-place concatenates); re-exports `encode`. |
| `__init__.py` | The replay of a spec's nodes while a template is traced: `computed` (on blocks of a loop or an epilogue), `indexed` (a loop's layout primitives on blocks), `per_row` (a row's values at their kinds' block shapes). |
| `indexing.rs` | Layout in loops, as XLA's indexing analysis at the grain of blocks: `plan` maps each value's dimensions to the output's or to one index and picks blocks every operand reads whole; `loop_fusion` asks it of every loop with layout. |
| `elementwise.rs` | Loops: `encode`, a fusion's body as `ops/elementwise/tpu`'s spec (plain, or `indexed` with layout). |
| `row.rs` | Rows: `fusible`, `kinds` (each value's kind, from the dataflow), `readable`, `fits` (what the diamonds pass asks) and `encode`, for `ops/row/tpu`. |
| `reduction.rs` | Reductions: `layout` (rows, columns or total), `reducible` and `encode` (prologue, reduce, epilogue), for `ops/reduce/tpu`. |
| `gemm.rs` | The dot's epilogue (`Epilogue::of`): its members' widths, extra inputs, nodes and outputs, for `ops/dot_general/tpu`. |
| `attention.rs` | Attention's forward and backward specs (`forward`, `backward`): each operand's view of its heads, the score chain, the outputs' layout (`written`) and the epilogue, for `ops/attention/tpu`; `graph_compiler/tpu` asks them at compile time. |

## Spec

`shape`, `inputs` (operands' dtypes), `broadcast` (whether each operand is
0-d), `nodes` (each `{name, params, args}`, `args` indexing the operands
then the nodes before), `result` (the value written), `output` (its dtype),
`blocks`, `window`. The launch's kernel name is `loop_<dtype>_<blocks>`; its
profiler name is the fusion's label (`exp → mul`).

A row's spec: `height` rows of `n`, `block` rows a step, `inputs` (each
`[dtype, kind]`), `nodes` (as a loop's, each with its value's `kind`),
`outputs` (each `[value, dtype, kind]`). A kind is `full` (a block of rows,
`(block, n)`), `row` (a value a row, `(block, 1)`), `lanes` (a value a
column, the same for every row: a norm's weight, `(1, n)`) or `scalar`. A
value's kind follows from how it is used (a reduction's operand is `full`,
a broadcast onto the last axis reads `lanes`, otherwise a broadcast reads
`row`), not its shape alone: `[n]` and `[rows]` look alike when `n` is the
row count. Its kernel name is `row_<rows>x<n>_<block>`.

## Invariants & gotchas

* A fusion has one output, and every operand has the output's shape or is
  0-d (read through a `broadcast_in_dim`); `encode` errors otherwise.
* A lone elementwise primitive is a one-node loop fusion: no per-primitive
  TPU kernel exists.
* Blocks are whole TPU tiles (`device::tpu::layout`): a shape with no such
  block errors.
* A row's operands and outputs must read as whole rows in their TPU layout
  (`readable`): `full` ones `[.., m, n]` with `m` whole tiles of rows (or
  one slab), `row` ones `[.., m, 1]` likewise, `lanes` ones `[n]`, scalars
  0-d. The diamonds pass leaves rows that don't to the other passes, as it
  does rows wider than a block of 2 MiB of 16 rows.

## Related

* [`../../graph_compiler/tpu/`](../../graph_compiler/tpu/) — `diamonds` (rows, before the dots take their epilogues) and `loop_fusion` (loops and reductions, after).
* [`../../ops/reduce/tpu/`](../../ops/reduce/tpu/) — the reduction template.
* [`../../graph_compiler/common/diamonds.rs`](../../graph_compiler/common/diamonds.rs) — the diamond matcher, shared with MPS.
* [`../../ops/tpu/`](../../ops/tpu/README.md) — routes a `loop` fusion's step here.
* [`../../ops/dot_general/tpu/`](../../ops/dot_general/) — the dot, whose epilogue uses `node` and `computed`.
