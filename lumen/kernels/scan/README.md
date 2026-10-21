# `lumen/kernels/scan`

Kernels for the `cumsum` primitive (inclusive cumulative sum along one
axis, optionally `reverse`, accumulating in `accum_dtype`). A graph
primitive, not an `Op`: on MPS a plan step reaches
[`mps/`](mps/README.md) through `kernels::mps::encode` (with scratch for split
scans); off MPS the plan evaluates it with `kernels/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` hook. |

## Subfolders

* [`mps/`](mps/README.md) — the Metal scan kernels and the host code that
  picks rows vs. lines and splits long scans into chunks.
