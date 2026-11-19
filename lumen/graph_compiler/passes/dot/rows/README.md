# rows

Passes that join a matmul with the normalization `row` fusion that reads its whole output rows, so the
product isn't written and read back (TPU) or is read once less (Triton). Each device has its own pass
type and its own slot in the pass list.

## Files

- `mod.rs` — the shared rewrite `dot_rows(graph, joins, fits, through)` and the `Joins` rule type.
- `tpu.rs` — `tpu::DotRows` (pass name `dot_rows`), TPU only.
- `triton.rs` — `triton::WideRows` (pass name `wide_rows`), Triton only.

CPU, MPS and CUDA get the identity pass in both slots. MPS joins a GEMM with its row inside the shared
fusion pass instead (`Backend::row_dot`, see `passes/fusion`).

## `dot_rows` (shared, used by TPU)

- Matches a `row` fusion whose operand at slot `at` is read whole by the row body (`joins(body, at)`)
  and comes from a plain dot: unbatched, a rank-2+ lhs contracted on its last axis, a 2-D rhs with one
  contracting axis. The dot's result must be read only by that row and not be a graph output.
- With `through`, one elementwise op may sit between the dot and the row (e.g. a residual add), as long
  as it has the dot's shape and its other readers come after the row. Its value becomes an extra output
  of the fusion (read back with `FusionOutput`) when something else reads it.
- Each dot joins at most one row; the joined body must pass the backend's `fits`.
- The result is one `dot_row` fusion whose inputs are the dot's operands, the linking op's other
  operands, then the row's other operands; its label joins the members' labels.

## TPU: `DotRows`

Slot after `fused_diamonds` (which makes the `row` fusions). Enabled with `normalization_diamonds`, and
it only rewrites when `contraction_epilogues` is also on. It calls `dot_rows` with a rule that accepts
inputs the TPU row kernel reads as `Kind::Full` rows, `fits` always true and `through` off.

## Triton: `WideRows`

Slot right after `fusion`. Enabled when `fuse`, `normalization_diamonds` and `linear_reordering` are all
on. It handles rows too wide for one GEMM tile (`partial_tile(n)` gives the column tile):

- Matches a `row` fusion fed by a GEMM (a plain dot or a GEMM fusion with its epilogue) whose output is
  read whole by the row, with tile-splittable last-axis reductions over the rows' shape
  (`diamonds::tile_reductions`).
- Replaces the GEMM with a `dot_partials` fusion that writes the row's operand plus each split
  reduction's partials over `tile`-column blocks (`[rows, n / tile]`), then the GEMM's own extra outputs.
- Reduces each partials output, reshapes it to the reduction's shape, and rebuilds the `row` fusion so it
  reads those totals as extra inputs instead of reducing the rows again.

The GEMM's values may be read elsewhere only after the row; its `FusionOutput` readers are renumbered
past the partials.
