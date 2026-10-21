# `lumen/ops/elementwise`

Elementwise loops on TPU: the template of a loop fusion (its body
generated, `codegen/tpu/elementwise.rs`). On MPS the loop is generated
whole (`codegen/mps/elementwise.rs`) on the prelude in `ops/mps`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `tpu` hook (`cfg(lumen_tpu_linked)`). |
| `tpu/mod.rs` | The loop's schedule: `blocks` (leading dimensions together, then whole tiles of the last two, smaller as streams add up). |
| `tpu/__init__.py` | `program(spec)`: a pipeline over blocks of the output replaying the spec's nodes (`codegen/tpu`'s `computed`, or `indexed` with each operand's own blocks); 0-d operands copied to VMEM once; an output each of `results`. |
