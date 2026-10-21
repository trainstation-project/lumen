# `lumen/kernels/layout/mps`

Metal kernels and host encoder for the layout primitives in graph plans on
MPS. Every primitive is expressed as one dtype-agnostic strided copy: output
element `i` reads the input at the offset of its index under per-output-dim
input strides. `encode(step, inputs, output, keep)` is called from the
primitive dispatch in [`lumen/kernels/mps/mod.rs`](../../mps/mod.rs).

## Files

| File | Contents |
| --- | --- |
| `kernels.metal` | `gather_<bytes>` (row-wise strided copy) and `transpose_<bytes>` (32x32 tiled transpose). |
| `mod.rs` | `encode` (strides per primitive) and `gather` (dim merging, kernel choice, grid). |

## Strides per primitive (`encode`)

| Primitive | Input strides per output dim |
| --- | --- |
| `Reshape` | a flat copy: shape `[numel]`, stride `[1]` |
| `BroadcastInDim` | 0 on new and size-1 input dims, else the input's contiguous stride |
| `Transpose { permutation }` | `xs[permutation[d]]` |
| `Slice { start_indices }` | the input's contiguous strides; the start offset is added to the source pointer |

`xs` are the input's contiguous strides: inputs are assumed dense.

## `gather` (host)

1. Drop size-1 dims and merge each dim into the previous one when
   `prev.stride == size * stride`.
2. If the innermost merged stride is > 1 and some dim `a` has stride 1, use
   `transpose_<bytes>`: dims `a` and the innermost `b` go through a
   `tile[32][33]` in threadgroup memory so reads (along `a`) and writes
   (along `b`) are both contiguous; other dims are batches.
   `Grid::Groups([ceil(a/32), ceil(b/32), batches])`, 16x16 threads, 2x2
   elements each.
3. Otherwise `gather_<bytes>`: thread `(x, y)` writes elements `x, x + X,
   ...` of output row `y`; the row's input offset comes from `offset_of32`
   over the outer dims. `Grid::Threads([ceil(inner / per_thread), rows, 1])`,
   `per_thread = 16 / width` when the inner stride is 0 or 1, else 1.

Kernels are instantiated with `FOR_BYTES` (1/2/4/8-byte elements), so any
dtype works.

## Invariants & gotchas

- Sizes, strides and offsets are 32-bit (`dims32_arg`, `offset_of32`,
  `uint` indices).
- `gather_<bytes>` here is a strided copy, not the `gather` primitive
  (that one is generated, `lumen/codegen/mps`).
- `kernels.metal` is concatenated after `lumen/kernels/mps/kernels.metal` by
  `build.rs` (`PRIMITIVE_KERNELS`), which provides `offset_of32` and `FOR_BYTES`.

## Related

- [`lumen/kernels/mps`](../../mps/) — `launch`, `Grid`, `BYTES_PER_THREAD`, `dims32_arg`.
- `lumen/tensor` — `contiguous_strides`.
