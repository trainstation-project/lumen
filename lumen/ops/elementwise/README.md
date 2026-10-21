# `lumen/ops/elementwise`

Kernels for the elementwise primitives (`add`, `sub`, `mul`, `div`, `max`,
`eq`, `lt`, `neg`, `exp`, `log`, `sqrt`, `tanh`, `logistic`, `cast`,
`select`, `full`) on TPU, one primitive a step, in lumen's Pallas.

On MPS there is no kernel here: every elementwise node runs in a generated
loop kernel (`lumen/codegen/mps`), alone as a one-node fusion
(`graph_compiler/mps/alone.rs`) when nothing fuses with it. The reference
interpreter (`lumen/ops/reference.rs`) evaluates these primitives itself.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `#[cfg(lumen_tpu_linked)] mod tpu;` hook. |

## Subfolders

[`tpu/`](tpu/) — the TPU kernel (`__init__.py`) and its encoder (`mod.rs`),
whose `node` also lowers a dot's epilogue.

## Related

- [`lumen/codegen/mps`](../../codegen/mps/README.md) — the loop kernels that run them on MPS.
- `lumen/graph/primitive.rs` — the `Primitive` enum.
