# scan

Kernels for `Cumsum`: an inclusive cumulative sum along one axis, optionally reversed, accumulating in the step's
`accum_dtype`. Each backend runs a fixed kernel; nothing fuses into a scan, so codegen only decides how the kernel
is split or tiled.

## Files

- `mod.rs` — the `ScanCodegen` trait: `scan(x, axis, reverse, accum_dtype)` returns the backend's `Code`. A backend
  whose kernel needs nothing generated returns `()`.
- `mps/` — Metal.
  - `kernel.metal`: simdgroup inclusive sums (Hillis-Steele over 32 lanes), a row kernel (`cumsum_rows_*`, a
    256-thread threadgroup scanning a contiguous row chunk) and a line kernel (`cumsum_lines_*`, one thread scanning
    a strided line chunk), instantiated per input and accumulator type. With `sum_only` they write each chunk's
    total to `partials`; otherwise they scan, starting from the scanned partials.
  - `mod.rs`: `ScanCodegen for Mps` generates nothing. `scan` views the tensor as `[a, n, b]` and picks the row
    kernel when the axis is contiguous and longer than `ROW_THREADS`, splitting lines into chunks until there is
    enough parallel work (`ROW_GROUPS`/`ROW_CHUNK`, `LINE_THREADS`/`LINE_CHUNK`). `encode` runs one pass, or three
    for a split scan: per-chunk totals into scratch, an in-place scan of those totals, then the chunked scan
    carrying each chunk's prefix. `scratch_bytes` (called from `codegen/utils/mps`) sizes the partials. `Cumsum` is
    the Kernel (`Config = ()`).
- `triton/` — Triton.
  - `kernel.py`: `scan_kernel`, a tile of lines a program, each line scanned `LANES` elements at a time with each
    chunk's total carried into the next. It reads `in0` and writes `out` through the generated `LOAD0` and `STORE`
    helpers, so it works on any layout.
  - `mod.rs`: `ScanCodegen for Triton` returns a `Scan` tiling (line length, inner stride, lanes, whether lines are
    rows, reverse, accumulator dtype), tiled the way a reduction over the axis is. `scan(step)` gives a cumsum
    step's tiling, which `codegen/utils/triton` turns into the `Template::Scan` constexprs, and the step runs under the Triton `Loop` Kernel.

## Conventions

- Scans are inclusive; `reverse` walks the axis from its end.
- The sum accumulates in `accum_dtype`, which may be wider than the input.
- MPS needs scratch only when it splits a scan.
