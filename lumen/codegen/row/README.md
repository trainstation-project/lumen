# row

Code generation for row fusions (normalization diamonds such as softmax, RMSNorm and LayerNorm). In a
row fusion, reductions run over each row's last axis and the rest of the body computes from the rows
and those per-row values. The diamonds pass (`graph_compiler/passes/diamonds`) forms these fusions, and
codegen recognizes one with `kinds::row_reductions` (Triton also checks the fusion name `row`).

## Files

- `mod.rs` — the `RowCodegen` trait: `row(body, by_value, options)` returns the backend's `Code`. The row pass traits (`NormalizationRows`, `FusedRows`, `FullRows`) live with their passes in `graph_compiler/passes`.
- `mps.rs` — Metal row kernels.
- `triton.rs` — Triton row kernels.
- `tpu/` — Pallas row program (`mod.rs`, `kernel.py`).

There is no CUDA row kernel. None of these backends has its own Kernel struct or configs. The steps run
on the generic per-backend kernels:

- MPS: `runtime::kernel::mps::Fusion`.
- Triton: `Loop` (`elementwise/triton`).
- TPU: `Pallas` (`codegen/devices/tpu`).

## Backends

**MPS (`mps.rs`).** `row_kernel` generates one kernel per body. It makes one pass over each row per
group of independent reductions, then a final pass that writes the outputs. `kept_values` keeps values
in registers between passes. When `simd_rows` holds (rows up to `SIMD_ROW` and SIMD-reducible
reductions), each row gets one SIMD group, with `SIMD_ROWS` rows per threadgroup. Otherwise each row
gets a threadgroup. `row_kernel` takes an optional GEMM `tile` so `dot_general/mps/row.rs` can reuse it
for rows fed by a fused matmul. `row_shape` and `row_groups` give the launch shape:
`graph_compiler/passes/fusion/mps.rs` reads `row_shape`, and `runtime::kernel::mps` reads `row_groups`.
`RowCodegen for Mps` takes the `CompilerConfig` as options. `codegen/devices/mps::kernel` calls it when a
body has row reductions.

**Triton (`triton.rs`).** `row` generates the kernel as a `reduction::triton::Reduction`:

- Rows up to `MOST_WHOLE_ROW` are held whole in registers, `BLOCK_SIZE // lanes` rows per program.
- Wider rows (`row_looped`) run one row per program, walked in `BLOCK_SIZE` chunks with one pass per
  reduction. A max and the sum of `exp(x - max)` share one online pass.
- A last pass writes the outputs.

`row_fits` says whether a body runs as one row kernel. `rows_type`, `row_count`, `width` and `row_groups`
describe its launch. `elementwise/triton` uses them. The per-reduction emit helpers (`operand`,
`identity`, `combined`, `whole`) are shared with `dot_general/triton/row.rs`.

**TPU (`tpu/`).** `kinds` classifies every value by how it lines up with the `rows x n` operand. The
`Kind`s are `Full`, `PerRow`, `Lanes` and `Scalar`. `None` means some value does not line up. These
functions are used outside this folder:

- `fusible` and `fits`: the diamonds and fusion passes call them to decide what may join a row fusion
  (dtypes of 32 bits or less, a minimal block within `ROW_BYTES`).
- `block_rows`: picks the rows per step.
- `kernel(step)`: wraps the spec as module `lumen.codegen.row.tpu.kernel` with `Rows::Split`.
- `rows`, `readable` and `row_node`: reused by `dot_general/tpu/row.rs` and the TPU dot-rows pass.

`kernel.py::program(spec)` streams blocks of whole rows through VMEM and replays the spec's nodes, each
tagged with its kind (`full`, `row`, `lanes`, `scalar`).

## Column partials

A row fusion can also carry column partials (`kinds::row_partials`, e.g. a norm weight's gradient).
These are per-column sums over groups of rows, written once per group. Each backend's `ROW_PARTIALS`
(`diamonds::Partials`) bounds the row width (`columns`) and the number of row groups (`groups`,
`fewest_groups`). The diamonds pass reads `ROW_PARTIALS` when it forms such fusions, so changing it
changes which fusions are formed.
