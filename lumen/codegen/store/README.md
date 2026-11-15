# store

How generated kernels address their loads and stores through a tensor's strides. The emitters (`codegen/emitter`) and every family generator on Metal and Triton share it, so each op's kernel reads strided inputs and writes outputs through views the same way. CUDA indexes through the strides kept by its emitter, and TPU through block specs, so neither has a file here.

## Files

- `mod.rs` — declares the two backends.
- `mps.rs` — Metal addressing:
  - `gather_index(idx, shape, strides)` gives the strided offset of a row-major flat index.
  - `load(k)` opens a call to input `k`'s `NAME_load{k}` helper.
  - `loads(source)` prepends a contiguous load helper for every `NAME_load{k}` that a source calls. `utils::mps::named` calls it when it names a kernel.
  - `strided_load` rewrites one of those helpers to read through given strides. The runtime (`runtime/kernel/mps.rs`) uses it to build strided variants of a kernel.
  - `written` generates the per-output `NAME_store{k}` helpers and their calls. Outputs that have a view are written at the view's offset and strides.
- `triton.rs` — Triton addressing:
  - `load(shape, strides)` and `store(shape, strides)` generate jit helpers that read or write flat logical index `i`.
  - `stored_at` gives a store's address expression.
  - `gather_index` gives the strided offset.

## Conventions

- `gather_index` merges adjacent dims that stay contiguous under the strides into one term, so fewer divide/modulo operations are emitted.
- Triton: a contiguous or one-element layout is addressed by the index itself. A tensor broadcast along every dim is loaded once and broadcast to the mask's shape. A store never broadcasts, because each output element is written at a distinct address.
- Kernels are written against the flat logical index. Only these helpers know the physical layout, so a kernel is regenerated per layout by swapping its helpers and keeping its body.
