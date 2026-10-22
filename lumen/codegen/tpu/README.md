# `lumen/codegen/tpu`

What TPU kernels generate from a fusion's body, as `codegen/mps` is to
Metal: each kind's spec (the body's nodes, its operands' views and blocks)
for a hand-written template in `kernels/<op>/tpu/` (a Pallas program, its
schedule), and the replay of those nodes while the template is traced
(`__init__.py`, the `Emitter` of a TPU: tracing inlines them, as Metal's
compiler inlines a functor). Templates: loops (`kernels/elementwise/tpu`), rows
(`kernels/row/tpu`), reductions (`kernels/reduce/tpu`), the dot
(`kernels/dot_general/tpu`) and attention (`kernels/attention/tpu`).

Fusions come from the TPU graph compiler: loops and reductions from
`loop_fusion` (`graph_compiler/tpu/fusion/`), rows from `diamonds`
(normalization diamonds, XLA's `SoftmaxRewriterTriton`), the dot's
epilogues from `fuse_dots`, attention's from `flash_attention`. Loops take
layout primitives too (`indexing.rs`): a broadcast, a reshape of unit
dimensions, a slice at whole blocks, a transpose of major dimensions and
iota become where each operand is read. Gathers (a lone one runs the
template in [`../../kernels/gather/`](../../kernels/gather/README.md)), update
slices and reshapes that move elements between tiles are not yet generated.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | What the kinds share: `fusible` (the primitives a loop fusion holds), `node` (a primitive as the spec's JSON) and `window` (an output's column window, for in-place concatenates); re-exports `encode`. |
| `__init__.py` | The replay of a spec's nodes while a template is traced: `computed` (on blocks of a loop or an epilogue), `indexed` (a loop's layout primitives on blocks), `per_row` (a row's values at their kinds' block shapes). |
| `indexing.rs` | Layout in loops, as XLA's indexing analysis at the grain of blocks: `plan` maps each value's dimensions to the output's or to one index and picks blocks every operand reads whole; `loop_fusion` asks it of every loop with layout. |
| `elementwise.rs` | Loops: `encode`, a fusion's body as `kernels/elementwise/tpu`'s spec (plain, or `indexed` with layout). |
| `row.rs` | Rows: `fusible`, `kinds` (each value's kind, from the dataflow), `readable`, `fits` (what the diamonds pass asks) and `encode`, for `kernels/row/tpu`. |
| `reduction.rs` | Reductions: `layout` (rows, columns or total), `reducible` and `encode` (prologue, reduce, epilogue), for `kernels/reduce/tpu`. |
| `gemm.rs` | The dot's epilogue (`Epilogue::of`): its members' widths, extra inputs, nodes and outputs, for `kernels/dot_general/tpu`. |
| `gemm_row.rs` | A dot and the row reading it (`dot_row`, from `dot_rows`): `encode`, the row's spec with the dot's product its first value, blocks of whole rows and of k, for `kernels/dot_general/tpu/row.py`. |
| `attention.rs` | Attention's forward and backward specs (`forward`, `backward`): each operand's view of its heads, the score chain, the outputs' layout (`written`) and the epilogue, for `kernels/attention/tpu`; `graph_compiler/tpu` asks them at compile time. |

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

A `dot_row`'s spec is a row's, its first value the dot's product (`[rows,
n]`, cast to the dot's dtype), with `k`, `transposed` (a `[n, k]` rhs), the
`operands`' and `value`'s dtypes and `blocks` (`[bm, bk]`): each step adds
`x[bm, bk] · y[bk, n]` into a float32 block of whole rows, the last runs the
row on it. Rows of up to 1024 blocks, k's as the dot's (`caps`), within VMEM.
Its kernel name is `matmul_row_<dtype>_<rows>x<n>x<k>_<bm>x<bk>`.

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

* [`../../graph_compiler/tpu/`](../../graph_compiler/tpu/) — `diamonds` and `dot_rows` (rows, and the dots they read whole, before the dots take their epilogues) and `loop_fusion` (loops and reductions, after).
* [`../../kernels/reduce/tpu/`](../../kernels/reduce/tpu/) — the reduction template.
* [`../../graph_compiler/common/diamonds.rs`](../../graph_compiler/common/diamonds.rs) — the diamond matcher, shared with MPS.
* [`../../kernels/tpu/`](../../kernels/tpu/README.md) — routes a `loop` fusion's step here.
* [`../../kernels/dot_general/tpu/`](../../kernels/dot_general/) — the dot, whose epilogue uses `node` and `computed`.
