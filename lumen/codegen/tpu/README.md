# `lumen/codegen/tpu`

TPU kernels generated from fused graphs, in lumen's Pallas
([`../../pallas/`](../../pallas/README.md)). The TPU graph compiler's
`loop_fusion` pass (`graph_compiler/tpu/loop_fusion.rs`) groups elementwise
chains into `Primitive::Fusion { name: "loop" }`; `encode` turns a fusion's
body into a spec, and the Python program (`__init__.py`) replays the body's
nodes on blocks of the output, which lumen's Pallas lowers to Mosaic.

Only loop fusions of elementwise primitives so far; layout primitives,
gathers and reductions are not yet generated.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | What the kinds share with the dot's epilogue: `fusible` (the primitives a loop fusion holds: elementwise, `full`, a 0-d operand's `broadcast_in_dim`, `Unary`/`Binary`), `node` (a primitive as the spec's JSON) and `window` (an output's column window, for in-place concatenates); re-exports `encode`. |
| `__init__.py` | `computed`: a spec's `nodes` replayed on blocks, for the generated kernels and the dot's epilogue. |
| `elementwise.rs` | Elementwise loops: the block choice (`blocks`) and `encode`, a fusion's body as its spec (only with `cfg(lumen_tpu_linked)`). |
| `elementwise.py` | `program(spec)`: the loop's kernel, a pipeline over blocks of the output (leading dimensions together, then the last two; or the elements) replaying the body; 0-d operands copied to VMEM once. |

## Spec

`shape`, `inputs` (operands' dtypes), `broadcast` (whether each operand is
0-d), `nodes` (each `{name, params, args}`, `args` indexing the operands
then the nodes before), `result` (the value written), `output` (its dtype),
`blocks`, `window`. The launch's kernel name is `loop_<dtype>_<blocks>`; its
profiler name is the fusion's label (`exp → mul`).

## Invariants & gotchas

* A fusion has one output, and every operand has the output's shape or is
  0-d (read through a `broadcast_in_dim`); `encode` errors otherwise.
* A lone elementwise primitive is a one-node loop fusion: no per-primitive
  TPU kernel exists.
* Blocks are whole TPU tiles (`device::tpu::layout`): a shape with no such
  block errors.

## Related

* [`../../graph_compiler/tpu/`](../../graph_compiler/tpu/) — `loop_fusion`, after the dots take their epilogues.
* [`../../ops/tpu/`](../../ops/tpu/README.md) — routes a `loop` fusion's step here.
* [`../../ops/dot_general/tpu/`](../../ops/dot_general/) — the dot, whose epilogue uses `node` and `computed`.
