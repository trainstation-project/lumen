# `lumen/codegen/tpu`

TPU kernels generated from fused graphs, in lumen's Pallas
([`../../pallas/`](../../pallas/README.md)). The TPU graph compiler's
`loop_fusion` pass (`graph_compiler/tpu/loop_fusion.rs`) groups elementwise
chains into `Primitive::Fusion { name: "loop" }`; `encode` turns a fusion's
body into a spec, and the Python program (`__init__.py`) replays the body's
nodes on blocks of the output, which lumen's Pallas lowers to Mosaic.

Two kinds so far: loops of elementwise primitives, and rows
(`Primitive::Fusion { name: "row" }`, from `graph_compiler/tpu/diamonds.rs`:
normalization diamonds, as XLA's `SoftmaxRewriterTriton`), whole rows a
block with their last-axis reductions. A third, reductions
(`Primitive::Fusion { name: "reduction" }`, from `loop_fusion`: a
reduction with the elementwise work producing its operand and consuming its
result), runs the hand-written template in `ops/reduce/tpu`, its prologue
and epilogue generated here, as MPS's reduction kernels instantiate
`ops/reduce/mps`. Loops take layout primitives too (`indexing.rs`): a
broadcast, a reshape of unit dimensions, a slice at whole blocks, a
transpose of major dimensions and iota become where each operand is read.
Gathers, update slices and reshapes that move elements between tiles are
not yet generated.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | What the kinds share with the dot's epilogue: `fusible` (the primitives a loop fusion holds: elementwise, `full`, a 0-d operand's `broadcast_in_dim`, `Unary`/`Binary`), `node` (a primitive as the spec's JSON) and `window` (an output's column window, for in-place concatenates); re-exports `encode`. |
| `__init__.py` | `computed`: a spec's `nodes` replayed on blocks, for the generated kernels and the dot's epilogue. |
| `indexing.rs` | Layout in loops, as XLA's indexing analysis at the grain of blocks: `plan` maps each value's dimensions to the output's (`Dim::Out`, at an offset for a slice) or to a single index (`Dim::One`, a broadcast or unit dimension), walking back from the output, and picks the output's blocks so every operand reads whole tiles at whole blocks; `layout` names the primitives it maps. `loop_fusion` asks it of every loop with layout. |
| `elementwise/mod.rs` | Elementwise loops: the block choice (`blocks`) and `encode`, a fusion's body as its spec (only with `cfg(lumen_tpu_linked)`). |
| `elementwise/__init__.py` | `program(spec)`: the loop's kernel, a pipeline over blocks of the output (leading dimensions together, then the last two; or the elements) replaying the body; 0-d operands copied to VMEM once. With layout (`indexed`), each operand its own `BlockSpec` from `plan`, and `replayed` runs the layout primitives on blocks (a broadcast of a block, a reshape of unit dimensions, a 2-D transpose, iota offset by the program ids). |
| `reduction.rs` | Reductions: `layout` (rows, columns or total, from the reduced axes; others are not run), `reducible` (what `loop_fusion` asks) and `encode` (the body as the template's spec: `prologue` nodes up to the reduced operand, `reduce`, `epilogue` nodes after it; a constant both read is copied into the epilogue). |
| `row/mod.rs` | Rows: `fusible` (what a row holds: the loop's primitives, reshapes, broadcasts, `reduce_sum`/`reduce_max`), `kinds` (each value's kind, from the dataflow), `readable` (whether an operand's TPU layout reads as whole rows), `fits` (what the diamonds pass asks) and `encode`. |
| `row/__init__.py` | `program(spec)`: the row's kernel, a pipeline over blocks of whole rows; `replayed` runs the body with each value at its kind's block shape. |

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
* [`../../graph_compiler/diamonds.rs`](../../graph_compiler/diamonds.rs) — the diamond matcher, shared with MPS.
* [`../../ops/tpu/`](../../ops/tpu/README.md) — routes a `loop` fusion's step here.
* [`../../ops/dot_general/tpu/`](../../ops/dot_general/) — the dot, whose epilogue uses `node` and `computed`.
