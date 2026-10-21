# `lumen/kernels/dynamic_slice/mps`

Metal kernels and host encoder for `dynamic_slice` / `dynamic_update_slice`
in graph plans on MPS, plus `copy`, a dtype-agnostic contiguous device copy.

`encode(step, inputs, output, keep)` is called from the primitive dispatch in
[`lumen/kernels/mps/mod.rs`](../../mps/mod.rs). `copy(src, dst, ty, keep, name)`
is also called by `dynamic_update_slice` here, by `scatter_add` in
[`lumen/kernels/gather/mps`](../../gather/mps/), by `lumen/graph_compiler/plan.rs`
(copying an operand that is not updated in place), by
`lumen/kernels/device_transfer/mps.rs` (`copy_h2d`) and by `Tensor::copy_` in
`lumen/tensor/tensor.rs`.

## Files

| File | Contents |
| --- | --- |
| `kernels.metal` | `dynamic_slice_*`, `dynamic_update_slice_*` and `dynamic_slice_copy_*` kernels; `block_offset` helper. |
| `mod.rs` | `encode` (picks kernel, binds index buffers, launches) and `copy`. |

## Kernels

All are instantiated with `FOR_BYTES`, i.e. by element size (1, 2, 4, 8
bytes as `uchar`..`ulong`), so one kernel serves every dtype of that size.

| Kernel | What it does | Grid |
| --- | --- | --- |
| `dynamic_slice_<bytes>_<i32\|i64>` | `out[j] = in[block_offset(j)]` | one thread per block element (`Grid::Threads([prod(slice_sizes),1,1])`) |
| `dynamic_update_slice_<bytes>_<i32\|i64>` | `out[block_offset(j)] = update[j]` | one thread per update element |
| `dynamic_slice_copy_<bytes>` | `out = in`, elementwise | `elementwise_grid` (16 bytes per thread) |

`block_offset` walks dimensions innermost-first; each start index is read
from its own scalar buffer (`I` = `int` or `long`) and clamped to
`[0, shape[d] - sizes[d]]` so the block stays inside the operand, as in XLA.

Buffer layout: 0 = source (operand, or the update), 1 = output, 2..9 = up to
8 start-index scalars, then `shape` (u64 array), `sizes` (u64 array), `rank`.

## Invariants & gotchas

- Rank is capped at `MAX_RANK = 8` (`DS_MAX_RANK` in Metal); `encode`
  returns an error for higher-rank operands.
- Unused index slots (rank < 8) are bound to the source pointer; they are
  never read.
- The index dtype in the kernel name comes from the first index input's
  dtype string (`i32` if there is none, i.e. rank 0).
- `dynamic_update_slice` writes in place into the output. If the output
  buffer is not the operand's buffer, `encode` first runs `copy` from the
  operand, then the update kernel.
- `kernels.metal` is concatenated after `lumen/kernels/mps/kernels.metal` by
  `build.rs` (`PRIMITIVE_KERNELS`), which provides `FOR_BYTES`,
  `ELEMENTWISE_ARGS` and `FOR_EACH_ELEMENT`. Instantiations sit behind
  `#ifndef TEMPLATES_ONLY`.

## Related

- [`lumen/kernels/mps`](../../mps/) — `launch`, `Grid`, `dims_arg`, `u32_arg`, prelude.
- [`lumen/kernels/gather/mps`](../../gather/mps/) — another indexed-access kernel family.
- `lumen/kernels/reference.rs` — reference semantics.
