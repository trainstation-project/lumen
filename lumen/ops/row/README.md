# `lumen/ops/row`

Row kernels on TPU: the template of a normalization diamond's fusion
(softmax, norms; its body generated, `codegen/tpu/row.rs`), whole rows a
block in VMEM. On MPS rows are generated whole (`codegen/mps/row.rs`).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `tpu` hook (`cfg(lumen_tpu_linked)`). |
| `tpu/mod.rs` | The row's schedule: `block_rows` (rows a block, within 2 MiB). |
| `tpu/__init__.py` | `program(spec)`: a pipeline over blocks of whole rows replaying the body (`codegen/tpu`'s `per_row`), each value at its kind's block shape. |
