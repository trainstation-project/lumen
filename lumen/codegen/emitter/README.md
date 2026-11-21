# emitter

The emitters write the statements that compute a fused body's values at an index, which is the core of every generated GPU kernel. Every family generator that turns graph nodes into source calls one: elementwise, reduction, row, layout/transpose, GEMM epilogues and prologues, and the attention score chain. TPU has no emitter here. Its Pallas programs rebuild nodes from JSON specs or `prims.bind` calls (see `codegen/devices/tpu`).

## Files

- `mod.rs` — the `Emit<'a>` trait: `new(body, by_value)` creates an emitter with nothing emitted yet, and `value(v, idx)` returns the local that holds value `v` at flat index expression `idx`, emitting any statements it needs first. It also defines `SwapsPairs` (`swap_pairs`) and `Gathers` (`take_along_axis`), for backends whose emitted code computes them.
- `cuda.rs` — CuTe DSL (Python) statements, one scalar element per thread.
- `triton.rs` — Triton (Python) statements over blocks of indices under a mask.
- `mps.rs` — Metal statements.

## How it works

Each `Emitter` walks back from the requested value through its producers. It memoizes every `(value, index)` pair and every index expression, so shared subexpressions are emitted once. Layout primitives (reshape, broadcast, transpose, slice, gather, iota and similar) become index arithmetic on the operand's index (`operand_index`), never materialized tensors; `SwapPairs` reads its operand at `idx ^ 1`, except that Triton swaps the adjacent lanes of a value computed from a register tile (`registers`, a GEMM epilogue's accumulator) with the `swap_pairs` helper of `codegen/devices/triton`. Constants are written bit-exactly through `codegen/devices` helpers. Random bits come from the `RandomCodegen` Philox expression in `codegen/random`. Callers read `lines` (plus `hoisted` on Metal) and splice them into their kernel.

## Per backend

- **cuda** — Loads take no mask because the kernel guards the index. Float add, subtract and multiply use `cute.math` with explicit nearest-even rounding, so no compiler stage contracts them into FMAs. Division and square roots are correctly rounded. `math` refers to `cute.math`, or to the fast approximations under `config.compiler.fast_math`. The emitter records `strides` per input, the `fragments` (inputs read at the walk's own index, which the elementwise kernel moves as vectors) and whether any read is `scattered` (out of row-major order).
- **triton** — Statements run under the current `mask`. `reduced` holds row reductions already computed by a row kernel. `tile` lets a GEMM row epilogue pick elements from a product tile it holds whole.
- **mps** — Values that do not change across the inner index (`invariant`) are computed once into `hoisted`, before the inner loop. `row_locals` holds values the enclosing kernel supplies. `wide` switches index arithmetic to 64-bit when any body tensor has more than `u32::MAX` elements.

## Conventions

- Programs run exactly as traced: emitters add no casts beyond the body's own, and rounding follows the traced dtypes.
- Input `k`'s loads go through the `load`/`LOAD{k}` helpers and the strided offsets in `codegen/store`, so one emitted body works for any input layout.
