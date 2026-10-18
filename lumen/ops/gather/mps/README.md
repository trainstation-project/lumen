# `lumen/ops/gather/mps`

Metal kernels and host encoder for `gather` and `scatter_add` in graph plans
on MPS. `encode(step, inputs, output, keep)` is called from the primitive
dispatch in [`lumen/ops/mps/mod.rs`](../../mps/mod.rs).

## Files

| File | Contents |
| --- | --- |
| `kernels.metal` | `gather_*`, `scatter_add_*` and `scatter_add_scan_*` kernel macros and instantiations. |
| `mod.rs` | `encode`: picks the kernel, copies the operand for `scatter_add`, sets the grid; `ROW` and `SCAN` constants. |

## Shapes

The operand `x` is viewed as `[outer, n, inner]` with `n = x.shape[axis]`
and `inner` the product of the dims after `axis`; the indices are flattened
to `m = indices.numel()` entries.

- `gather`: `out[o, t, r] = x[o, index[t], r]`.
- `scatter_add`: `out = x`, then `out[o, index[t], r] += updates[o, t, r]`.
  Inputs are `(x, indices, updates)`; kernel buffers are
  `(updates, indices, out)`.

Args for every kernel: `n`, `m`, `inner` (at least 1), `total`, all u64.

## Kernels

| Kernel | Dtypes | Strategy / grid |
| --- | --- | --- |
| `gather_<bytes>_<i32\|i64>` | any, by element size (`FOR_BYTES`) | one thread per output element; `Grid::Threads([min(total, ROW), ceil(total / ROW), 1])` with `ROW = 2^20` |
| `scatter_add_<dt>_<i32\|i64>` | `FOR_NUMERIC` (no bool) | one thread per `(o, r)` (`total = numel / n`), looping over all `m` indices in order; same 2-D grid |
| `scatter_add_scan_<dt>_<i32\|i64>` | `FOR_NUMERIC` | used when `n * m <= SCAN` (`2^16`): one thread per output element, `Grid::Groups([ceil(total / 256), 1, 1])` (16x16 threadgroups); indices staged 256 at a time in threadgroup memory, each thread adds the updates whose index equals its row; when `inner % 32 == 0` a `simd_ballot` tests 32 indices at once |

Index dtype suffix is `i32` for `DType::I32`, otherwise `i64`.

## Invariants & gotchas

- `scatter_add` writes each output element from a single thread in index
  order: no atomics, deterministic sums.
- If the output buffer is not the operand's, `encode` first copies the
  operand with [`dynamic_slice::mps::copy`](../../dynamic_slice/mps/mod.rs).
- `scatter_add` on `bool` returns an error ("booleans have no sum").
- Indices are read as given (`ulong(index[t])`); there is no clamping or
  bounds check.
- `total == 0` or `n == 0`: nothing is launched (`profiler::clear_next_launch`).
- The 2-D thread grid exists because Metal grid dimensions are 32-bit.
- `kernels.metal` is concatenated after `lumen/ops/mps/kernels.metal` by
  `build.rs` (`PRIMITIVE_KERNELS`), which provides `Add` and the `FOR_*`
  macros; instantiations sit behind `#ifndef TEMPLATES_ONLY`.

## Related

- [`lumen/ops/dynamic_slice/mps`](../../dynamic_slice/mps/) — `copy`.
- [`lumen/ops/mps`](../../mps/) — `launch`, `Grid`, `u64_arg`; `shim.mm` sets the 16x16 threadgroup for `Grid::Groups`.
- `lumen/ops/reference.rs` — reference semantics.
