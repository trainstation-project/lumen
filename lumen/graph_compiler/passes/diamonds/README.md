# passes/diamonds

Normalization rows ("diamonds"): a value combined with a broadcast of its own last-axis reduction, such as
softmax, RMS norm or layer norm. A backend can run each one as a single row kernel. This folder finds those
rows. On TPU it fuses them directly; on MPS and Triton it hands them to `fusion`.

## Pattern

- A diamond is an elementwise node `side op broadcast(reduce(x))`, where `side` and `x` come from the same
  value through trivial ops. For `Add`, `Mul` and `Max` either operand may be the side. Single-use
  elementwise producers of that value are absorbed too.
- `diamonds(graph, fusible)` builds one `Row` per diamond, then grows it:
  - A diamond over the root of an earlier row that only it reads absorbs that row (chained norms).
  - `grown` extends rows backwards over same-shaped producers, per-row values and earlier rows, so a
    pattern with several reductions becomes one row.
  - `consumed` moves a row's root forward to a later elementwise consumer whose cheap inputs reach the row.
- Only nodes the backend's `Fusible` accepts go into a row.
- `Row` lists the root, the last-axis reductions, the other inner nodes, the nodes read outside the row,
  attached column partial sums, and `statistics` (a matmul's rows plus per-row values of one reduction of
  them, CODA's statistics).

## Passes (both named `normalization_diamonds`)

- `Diamonds` - finds the rows, and with `partials` splits column sums over the same rows into a per-group
  partial inside the row kernel plus a final sum over groups (bounded by the backend's `Partials`:
  widest row, most and fewest groups). It stores the rows in the module (`RowsModule::rows_mut`, i.e.
  `DeviceModule::rows`). It runs after `dynamic_update_slices`, and `fusion` consumes and clears the rows.
- `FusedDiamonds` - finds the rows and rewrites each one whose body the backend `fits` into a `row`
  fusion (`fused`), with extra outputs through `FusionOutput`. A row is emitted only when it is convex
  and sealed. It runs after `canonicalize_dots` and before TPU's `dot_rows`.

## Files

- `mod.rs` - `Row`, `Finder`, `diamonds`, `fused`, `partials`, both passes, and helpers codegen uses:
  `depends_on`, `softmax_pair` (`sum(exp(x - max(x)))` over the same axes) and `tile_reductions` (which
  leading row reductions a tiled pass may take partially; used by Triton `wide_rows` and the MPS GEMM row
  kernels).
- `mps.rs` - `Diamonds` with `fusion::mps::fusible` and `codegen::row::mps::ROW_PARTIALS`, when `fuse`
  and `normalization_diamonds` are on.
- `triton.rs` - `Diamonds` with `fusion::triton::fusible` and `codegen::row::triton::ROW_PARTIALS`, under
  the same flags, finding the same rows as MPS.
- `tpu.rs` - `FusedDiamonds` with `codegen::row::tpu::fusible` and `fits`, when `normalization_diamonds` is
  on.
- CUDA and CPU: identity for both passes.
