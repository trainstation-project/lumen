# `lumen/ops/reduce/mps`

Metal kernels for `reduce_sum` / `reduce_max` and the host code that picks a
layout, kernel, grid and (for long reductions) a two-pass split. Reached
from `ops::mps::encode` (`encode`, with scratch). `graph_compiler/mps` calls
`encode_reduction` for fused reductions, passing its own generated kernel
names (`fused` for the first pass, `last` for the second) plus extra
buffers and scalar args; `codegen/mps` uses `layout`, `splits` and
`REDUCE_THREADS` to generate matching kernels from the templates here.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `encode`, `encode_reduction`, `Layout` / `layout`, `normalize`, `split`, `scratch_bytes`, `splits`, `accum_dtype`, `kernel_dtype`, `REDUCE_THREADS`. |
| `kernels.metal` | Templates `reduce`, `reduce_grouped`, `reduce_rows`, `reduce_cols` and their kernel instantiations (`REDUCE`, `GROUPED`, `ROWS`, `COLS`). |

## Layouts

`normalize` first drops size-1 dims and merges adjacent dims that are both
reduced or both kept. Then `layout` picks:

| Layout | When | Kernel | Grid |
| --- | --- | --- | --- |
| `Rows` | reduced dims consecutive and last (`b == 1`), with ≥ 256 lanes of work (or input over `u32::MAX`) | `<op>_rows_<dtype>` | `Groups([chunks, a, 1])`: 256 threads a row chunk, 4-way unrolled, tree-reduced in threadgroup memory |
| `Cols` | reduced dims consecutive, kept dims after them (`b > 1`) | `<op>_cols_<dtype>` | `Threads([a·chunks·b])`: a thread a column chunk, 4-way unrolled |
| `Grouped` | short rows, or non-consecutive reduced dims; input ≤ `u32::MAX` elements | `<op>_grouped_<dtype>` | `Groups([ceil(outputs / (256/lanes))])`: `lanes` (a power of two ≤ 256, ~32 bytes of input each) threads per output |
| `Generic` | otherwise | `<op>_<dtype>` | `Threads([outputs])`: a thread an output, strided `offset_of` |

Here the input is viewed as `[a, count, b]` (kept, reduced, kept).

## Split reductions

For `Rows`/`Cols`, `split` cuts `count` into `chunks` when there would be
too little parallelism (rows: fewer than 256 rows, chunks ≥ 4096 elements;
cols: fewer than 2¹⁶ outputs, chunks ≥ 64). Then two launches:
the first writes partials `[a, chunks, b]` of the accumulator dtype to
**scratch**, the second reduces them with `<op>_<layout>_<accum>`.
`scratch_bytes` returns `a · chunks · b · sizeof(accum)` (0 if unsplit); a
split step without scratch errors.

## Dtypes

* `<dtype>` is `kernel_dtype(x, accum)`: the input dtype, or `<x>_<accum>`
  when they differ. Only `f16_f32` and `bf16_f32` exist for `reduce_sum`
  (half/bfloat summed and output in float).
* `reduce_sum` covers numeric dtypes; `reduce_max` also covers `bool`.
* `reduce_max`'s init/identity is `-inf` for floats, the lowest value for
  signed ints, 0 / `false` otherwise; `max_op` propagates NaN.

## Invariants & gotchas

* Every output is reduced in a fixed order (per thread, then a fixed tree,
  then the partials in order): results are deterministic run to run.
* Threadgroup kernels assume 256 threads (`REDUCE_THREADS`, matching
  `Grid::Groups`' 16 × 16).
* Instantiations sit under `#ifndef TEMPLATES_ONLY`; fusion kernels include
  the templates only.

## Related

* [`../../mps/README.md`](../../mps/README.md) — launch, `Max`/`Add` functors.
* [`../../scan/mps/README.md`](../../scan/mps/README.md) — same splitting scheme for `cumsum`.
* `lumen/codegen/mps/mod.rs` — fused reduction kernels.
