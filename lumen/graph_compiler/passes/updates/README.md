# updates

`Updates` (pass name `updates`) fuses chains of `dynamic_update_slice`s with compile-time start indices
into one in-place `updates` fusion (`graph::UPDATES`): one kernel writing every update into the operand.
It runs after `dot_rows` and before `diamonds`.

## Files

- `mod.rs` — the pass and helpers codegen reads (`starts`, `overlaps`).
- `mps.rs` — MPS runs it with `fuse`, chains of 2 to 29 updates (Metal binds 31 buffers, two of them the
  operand and the output). Built only when MPS is linked.
- `triton.rs` — Triton runs it with `fuse`, chains of 2 or more updates.
- `tpu.rs` — TPU runs it always, chains of 1 or more updates.

CPU and CUDA get the identity pass.

## Rewrite

- An update qualifies when every start index is a scalar `i32` integer fill (a compile-time constant).
- An update extends the chain of the update producing its operand when that operand is read only by it, is
  not a graph output, and both updates have the same update type. A chain longer than `most` continues as
  a new chain.
- Each chain of `least` to `most` updates becomes one `updates` fusion, emitted at the chain's last node.
  Its inputs are the operand and each update; the start indices are rebuilt as constants inside the body.

## Used by codegen

- `starts(body)` lists each update of an `updates` body with its constant start indices, in order.
- `overlaps(x, update, starts)` gives each update's start indices clamped in bounds and the later updates
  whose region overlaps it. A kernel writing all updates at once skips those elements, so the last update
  to write an element wins, as when they run in order. The MPS and Triton dynamic-slice codegen use both.

`fusion::mps` keeps `updates` fusions out of horizontal fusion.
