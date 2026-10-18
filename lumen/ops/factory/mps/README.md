# `lumen/ops/factory/mps`

Metal kernels and host encoder for `full`, `iota` and `random_bits` in graph
plans on MPS. `encode(step, inputs, output, keep)` is called from the
primitive dispatch in [`lumen/ops/mps/mod.rs`](../../mps/mod.rs).

## Files

| File | Contents |
| --- | --- |
| `kernels.metal` | `fill_<bytes>`, `iota_<dt>` and `random_bits` kernels. |
| `mod.rs` | `encode`: one arm per primitive, building args and grid, then `launch_step`. |

## Kernels

| Primitive | Kernel | Dtypes | Grid / args |
| --- | --- | --- | --- |
| `Full { fill_value }` | `fill_<1\|2\|4\|8>` | any, by element size (`FOR_BYTES`) | `elementwise_grid`; args: the value converted to the output dtype (`element_arg`), `n` (u32) |
| `Iota { dimension }` | `iota_<dt>` | `FOR_NUMERIC` (no bool) | `Grid::Threads([inner, shape[dimension], outer])`; no args |
| `RandomBits { offset }` | `random_bits` | `uint32` output | `elementwise_grid`; args: `offset` (u64), `n` (u32) |

- `iota`: the value written is the thread's y coordinate (`from_int<T>(gid.y)`),
  the index along `dimension`; x covers the dimensions after it, z those before.
- `random_bits`: element j is `philox_bits(*seed, *start + offset + j)`.
  `seed` and `start` are one-element `ulong` buffers (inputs 0 and 1), so a
  fusion can take them by value; `offset` is a constant from the primitive.

## Invariants & gotchas

- Kernel `fill_<bytes>` here (e.g. `fill_4`) is unrelated to the eager
  `fill_u32` / `fill_strided_u32` kernels in
  [`lumen/ops/fill/mps`](../../fill/mps/); they live in different Metal
  libraries.
- `n` is `u32`; the iota grid uses 32-bit coordinates per dimension.
- `kernels.metal` is concatenated after `lumen/ops/mps/kernels.metal` by
  `build.rs` (`PRIMITIVE_KERNELS`), which defines `FOR_BYTES`,
  `FOR_NUMERIC`, `from_int`, `philox_bits`, `ELEMENTWISE_ARGS` and
  `FOR_EACH_ELEMENT`.

## Related

- [`lumen/ops/mps`](../../mps/) — `launch_step`, `element_arg`, `elementwise_grid`.
- `lumen/ops/reference.rs` — reference semantics for the same primitives.
