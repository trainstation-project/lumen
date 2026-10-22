# `lumen/graph_compiler/tpu`

The TPU pipeline: the passes that turn a program's graph into fusions the
TPU kernels run (`kernels/tpu`, `codegen/tpu`), then the plan. Laid out as the
MPS pipeline is (the forward attention pass shared, `../common/flash_attention.rs`) ([`../mps/`](../mps/README.md)): a file per pass, loop
fusion a folder per phase.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `TpuModule` (its `Attentions`: epilogues reading scalars, the raw output with heads merged), the TPU layout's sizes (`LAYOUT`), `compile` (the pipeline below, then `Plan::compile_with`). |
| `flash_attention_backward.rs` | Pass `flash_attention_backward`: attention's backward as its `dq` and `dkv` fusions; their outputs laid out for the kernels (`merged_outputs`, `reshaped_outputs`), Δ computed inside them (`delta_inside`), and a check that every attention runs (`attentions`). |
| `diamonds.rs` | Pass `normalization_diamonds`: each normalization diamond (`graph_compiler/common/diamonds.rs`) the row kernel can run one `row` fusion. |
| `updates.rs` | Pass `dynamic_update_slices` (`Updates`): each chain of `dynamic_update_slice`s of one update shape at constant starts, each the only reader of the last, one `dynamic_update_slices` fusion (`kernels/dynamic_slice/tpu`); the plan writes it into its operand as it does an update. |
| `place_concats.rs` | Pass `place_concats`: concatenates along the last dimension whose operands' kernels write them in place (`placeable`). |
| `fusion/mod.rs` | Pass `loop_fusion` (`LoopFusion`); re-exports `FuseDots`. |
| `fusion/dot_epilogues.rs` | Pass `fuse_dots` (`FuseDots`): a dot and its column or row slices and elementwise consumers one `dot_general` fusion, the dot kernel's epilogue (members, together, in-place outputs). |
| `fusion/groups.rs` | Loop fusion's groups: each fusible node joins the group of its readers (`groups`); a reduction one a group, layout only where `codegen/tpu/indexing.rs` plans it (`planned`). |
| `fusion/emit.rs` | Each group as a `loop` or `reduction` fusion (`members`, `body`, `loop_fusion`). |
| `fusion/horizontal.rs` | Pass `horizontal_fusion` (`HorizontalFusion`): independent small elementwise loops of one shape as one multi-output loop (`../common/horizontal.rs`, with the TPU's `Target`: up to 1M elements, 32 streams, none a concatenate writes in place). A TPU kernel costs ~2–8 µs besides its bytes. |

## Pipeline

| # | Pass | When | File |
| --- | --- | --- | --- |
| 1 | `merge_dots_backward` | `merge_dots` | `../common/merge_dots_backward.rs` |
| 2 | `flash_attention`, `flash_attention_backward` | `flash_attention` | `../common/flash_attention.rs`, `flash_attention_backward.rs` |
| 3 | `fold_dot_reshapes`, `simplify`, `dot_strength` | always | `../common/` |
| 4 | `merge_dots` | `merge_dots` | `../common/merge_dots.rs` |
| 5 | `normalization_diamonds` | `normalization_diamonds` | `diamonds.rs` |
| 6 | `dynamic_update_slices` | always | `updates.rs` |
| 7 | `fuse_dots` | always (epilogues with `contraction_epilogues`) | `fusion/dot_epilogues.rs` |
| 8 | `place_concats` | always | `place_concats.rs` |
| 9 | `loop_fusion` | always | `fusion/` |
| 10 | `horizontal_fusion` | `horizontal_fusion` | `fusion/horizontal.rs` |

The frontend (`cse`, `simplify`) runs before, for every device; the plan's
stages (scheduling, rematerialization, buffers) after.

## Related

* [`../../codegen/tpu/`](../../codegen/tpu/README.md) — the generated kernels of `loop`, `row` and `reduction` fusions.
* [`../../kernels/tpu/`](../../kernels/tpu/README.md) — routes each step to its kernel.
