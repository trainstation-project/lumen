# `lumen/ops/scan/mps`

Metal kernels for `cumsum` and the host code that launches them. Reached
only from `ops::mps::encode` (and `ops::mps::scratch_bytes` for the
workspace). The operand is viewed as `[a, n, b]`: `a · b` lines of `n`
elements along the scanned axis, each scanned from its start, or from its
end when `reverse`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `scan` (shape split and chunking), `scratch_bytes`, `encode`; the tuning constants `ROW_THREADS`, `ROW_GROUPS`, `ROW_CHUNK`, `LINE_THREADS`, `LINE_CHUNK`. |
| `kernels.metal` | `shuffle_up`, `simd_inclusive_sum` (Hillis–Steele over a SIMD group), `carried`, the `cumsum_rows` / `cumsum_lines` templates and the `CUMSUM` kernel pair per dtype. |

## Kernel variants

| Variant | When | Grid | How |
| --- | --- | --- | --- |
| `cumsum_rows_<dtype>` | `b == 1` and `n > 64` | `Groups([chunks, a, 1])` | a 256-thread threadgroup per row chunk, in blocks of 256 × 4 elements: each thread sums 4 consecutive elements, a SIMD-group scan, then a scan of the SIMD groups' totals in threadgroup memory, gives each thread its prefix |
| `cumsum_lines_<dtype>` | otherwise | `Threads([a · b · chunks])` | a thread per line chunk, sequential; neighbouring threads take neighbouring columns |

`<dtype>` is `kernel_dtype(x, accum)` (from `reduce/mps`): the input dtype,
or `f16_f32` / `bf16_f32` for half/bfloat accumulating (and output) in float.
All numeric dtypes are instantiated; integers wrap.

## Split scans

`scan` splits each line into `chunks` while there is too little
parallelism: rows when fewer than `ROW_GROUPS` (256) threadgroups would
run, chunks of at least `ROW_CHUNK` (4096); lines when fewer than
`LINE_THREADS` (16384) threads would run, chunks of at least `LINE_CHUNK`
(64). With `chunks > 1`, three launches, all in a fixed order:

1. the same kernel with `sum_only = 1` writes each chunk's sum to the
   partials (`[lines, chunks]` of the accumulator dtype, in **scratch**);
2. `cumsum_lines_<accum>` scans the partials in place (one line of chunks
   per thread);
3. the kernel again with `sum_only = 0`, each chunk starting from the
   scanned sum of the chunks before it (`carried`).

`scratch_bytes` = `a · b · chunks · sizeof(accum)`, 0 when unsplit; a split
scan without scratch errors.

## Invariants & gotchas

* Deterministic: every output is computed in a fixed order.
* `encode` errors if `a · b · chunks` exceeds `u32::MAX` (thread ids are
  32-bit).
* `cumsum_rows` assumes 256-thread threadgroups (`Grid::Groups`' 16 × 16).
* This file is not in the fusion prelude (`PRELUDE` in
  `graph_compiler/mps/mod.rs`), unlike reduce and dot_general.

## Related

* [`../../mps/README.md`](../../mps/README.md) — launch and the shared prelude.
* [`../../reduce/mps/README.md`](../../reduce/mps/README.md) — the analogous split for reductions; `kernel_dtype`.
