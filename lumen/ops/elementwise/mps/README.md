# `lumen/ops/elementwise/mps`

One Metal kernel per (primitive, dtype) for the unfused elementwise
primitives in graph plans on MPS. The op is fixed at compile time by
applying a functor from [`lumen/ops/mps/kernels.metal`](../../mps/kernels.metal).

`encode(step, inputs, output, keep)` is called from the primitive dispatch in
[`lumen/ops/mps/mod.rs`](../../mps/mod.rs) for `Add | Sub | Mul | Div | Max |
Eq | Lt | Neg | Exp | Log | Sqrt | Tanh | Logistic | Cast | Select`.

## Files

| File | Contents |
| --- | --- |
| `kernels.metal` | `BINARY`, `NEG`, `UNARY`, `CONVERT` and `SELECT` kernel macros and their instantiations. |
| `mod.rs` | `encode`: builds the kernel name and grid, then `launch_step`. |

## Kernels and dtypes

| Kernels | Dtypes (macro) | Output |
| --- | --- | --- |
| `add_`, `sub_`, `mul_`, `div_<dt>` | `FOR_NUMERIC`: u8..u64, i8..i64, f16, bf16, f32 | same dtype |
| `max_<dt>` | `FOR_ALL` (numeric + bool) | same dtype |
| `eq_`, `lt_<dt>` | `FOR_ALL` | `bool` |
| `neg_<dt>` | `FOR_NUMERIC` | same dtype |
| `exp_`, `log_`, `sqrt_`, `tanh_`, `logistic_<dt>` | `FOR_MATH_FLOAT`: f16, f32 only | same dtype |
| `convert_<src>_<dst>` | every pair of `FOR_ALL` dtypes | `<dst>` |
| `select_<bytes>` | by element size 1/2/4/8 (dtype-agnostic) | case dtype |

Kernel name in `encode`: `Cast` -> `convert_{in}_{out}`; `Select` ->
`select_{out.size_of()}`; anything else -> `{primitive.name()}_{in dtype}`.
`<dt>` is the dtype's `Display` name (`f32`, `bf16`, `i64`, ...).

## Grid

`elementwise_grid(n, dtype)`: `ceil(n / (16 / size_of))` threads in x. Each
thread handles up to `per_thread<T>()` elements via `FOR_EACH_ELEMENT`,
striding by the grid size (`j = i, i + threads, ...`). Args are just `n`
(u32). Buffers are the step inputs in order, then the output
(`launch_step`); `select` takes `pred, a, b, out`.

## Invariants & gotchas

- `n` is passed as `u32`; element indices in the kernels are `uint`.
- There are no f64 kernels, and no bf16 kernels for the transcendental
  unaries (`FOR_MATH_FLOAT` omits bf16).
- `select`'s grid is sized by the output dtype, comparisons' by the input
  dtype ("threads take elements of the input").
- `kernels.metal` is concatenated after `lumen/ops/mps/kernels.metal` by
  `build.rs` (`PRIMITIVE_KERNELS`); it relies on its macros (`FOR_*`,
  `ELEMENTWISE_ARGS`, `FOR_EACH_ELEMENT`) and `convert_value`.

## Related

- [`lumen/ops/mps`](../../mps/) — `launch_step`, `elementwise_grid`, functors.
- `lumen/graph_compiler/mps` — fused elementwise kernels.
