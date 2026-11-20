# scatter_add

Kernels for `ScatterAdd`: rows of `updates` added into an operand at `indices` along one axis, with inputs
`[operand, indices, updates]`. The output first receives a copy of the operand, then the updates are added in.
Nothing fuses into a scatter-add, so there is no codegen trait and no generated code, only fixed kernels and
their launch.

## Files

- `mod.rs` — declares the backends.
- `mps/` — Metal.
  - `kernel.metal`: atomic adds (native, or compare-and-swap on the 32-bit word holding a 16-bit value, since
    Metal has no 16-bit atomics), and the deterministic path: `scatter_add_count_*` (updates per position),
    `scatter_add_place_*` (each update's number into its position's segment of `order`) and `scatter_add_rows_*`
    (a threadgroup per position, column block and outer index sorts its segment and adds the updates in update
    order, a thread a column). A position hit more often than a segment holds walks all indices in order.
    Kernels exist for int32 and int64 indices.
  - `mod.rs`: `encode` runs the copy then either the atomic kernel (float32, float16, bfloat16, int32, uint32 when
    not `deterministic`, a thread an element) or the count, prefix-sum, place and rows passes. `scratch_bytes`
    sizes the counts, starts and order (`2n + 1 + m` u32s) and is used by `codegen/utils/mps`. `ScatterAdd` is the
    Kernel (`Config = ()`).
- `triton/` — Triton.
  - `mod.rs`: `atomic_add(dtype)`, whether `tl.atomic_add` adds the dtype (else the sorted path runs).
  - `kernel.py`: views the operand as `[outer, LENGTH, AFTER]` and updates as `[outer, PICKS, AFTER]`, both
    contiguous. `scatter_add_atomic_kernel` adds each update element atomically; the deterministic path is
    `count`, `starts`, `place`, `sort` (a counting sort of picks by position) then `rows` (a program per block of a
    position's entries adding its updates in pick order, as MPS does). Every kernel takes the same constexprs
    (`N` plus `OUTER`, `LENGTH`, `AFTER`, `PICKS`).
  - `config.rs`: fixed `LoopConfig`s per kernel (`ELEMENTS`, `STARTS`, `SEGMENT`, and `rows` for the rows kernel's
    block size). None is autotuned: the tuner's zeroed indices would send every update to one position.
  - `launch.rs`: the `Scatter` enum naming each kernel, and `run`, which launches the atomic kernel or the sort
    then rows kernels. It is called from the Triton `Loop` Kernel (`elementwise/triton`) for steps codegen gave a
    `Template::ScatterAdd`.

## Conventions

- `deterministic` (`lumen.config`) selects the sorted path, so duplicate indices sum in update order as the CPU
  does. Without it, dtypes that have atomic adds use them; on Triton, dtypes `tl.atomic_add` lacks (bool, float8,
  narrow integers) always take the sorted path.
- Updates and indices are read contiguous; scatter-adds take no strided inputs.
