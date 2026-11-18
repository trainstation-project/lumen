# place_concats

In-place concatenation: the kernels producing a `Concatenate`'s operands write them straight into their
slot of the result, so the concatenation copies nothing. Two slots use it. `place_concats` marks
concatenations (`in_place = true`) before `fusion`, right after the dot passes; `concats` (MPS only) runs
after `fusion`, `place_views` and `alone` and unmarks the ones fusion made invalid.

## Files

- `mod.rs` — the `Slots` trait, `worth` (the rule MPS and Triton share), `in_place_concats`, the `PlaceConcats`
  pass, and `Concats`, the slot type of the post-fusion check.
- `mps.rs` — MPS: `PlaceConcats` with `fuse`; `Concats` always; its slot rule `slot_writes`. Built only when
  MPS is linked.
- `triton.rs` — Triton: `PlaceConcats` with `fuse`; a slot is writable when `fusion::triton::slot_writes`
  accepts the producer (a matmul, or a fusible elementwise or copying layout op).
- `tpu.rs` — TPU: `PlaceConcats` always, with `fusion::tpu::slot_writes` (the concat joins its last axis,
  the operand fills whole lanes, and its producer writes a strided slice), and every operand worth placing.

CPU and CUDA get the identity pass in both slots; Triton and TPU have no `Concats`.

## Rule

`in_place_concats(graph, writes, worth)` returns the live concatenations where every operand:

- is read only by that concatenation (ignoring `FusionOutput` readers), exactly once, and is no program
  output;
- is made by a kernel `writes(concat, k, producer, index)` accepts; for an operand that is output `index`
  of a fusion, the producer is that fusion;

and at least one operand is `worth` placing. `Slots::worth` defaults to `worth`, which MPS and Triton share
(only `writes`, what each backend's kernels can write through a slot's strides, differs): an operand is worth
placing when a kernel of its own makes it (a matmul, attention's backward, a fusion), whose output written into
its slot needs no copy. A cheap elementwise or layout producer is not: fusion builds the concatenation inside a
kernel that runs anyway, its readers' (rotary's `cat([-x2, x1])` before `* sin`) or its other operands'. TPU
places whenever it can.

## MPS rules

`slot_writes` accepts the attention backward kernel (except its atomically accumulated dQ), a fusion whose
body writes views, and, before fusion, a fusible layout or elementwise node. Since a matmul writes no slot on
MPS, its concatenations placed are those gathering attention gradients.

`Concats` re-runs `in_place_concats` with the post-fusion rules, clears `in_place` on concatenations no
longer valid, and wraps each such concatenate in its own elementwise fusion kernel (`alone::wrapped`).

## Users

`passes/fusion` treats in-place concatenations as unfusible, and `dot/fuse/tpu.rs` calls
`in_place_concats` to avoid building a placed concat inside a dot fusion.
