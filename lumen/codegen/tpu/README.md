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
| `mod.rs` | `fusible` (the primitives a loop fusion holds: elementwise, `full`, a 0-d operand's `broadcast_in_dim`, `Unary`/`Binary`), `node` (a primitive as the spec's JSON, also the dot's epilogue), the block choice (`blocks`), output windows (`window`, for in-place concatenates) and `encode` (only with `cfg(lumen_tpu_linked)`). |
| `__init__.py` | `program(spec)`: the generated kernel, a pipeline over blocks of the output (leading dimensions together, then the last two; or the elements) whose body replays `nodes` (`computed`, which the dot's epilogue uses too); 0-d operands copied to VMEM once. |

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
