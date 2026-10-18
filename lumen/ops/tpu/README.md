# `lumen/ops/tpu`

TPU kernels and how they run. A plan step on a TPU (`Plan::execute`, the
executor `Device::Tpu`) goes to `encode`, which routes on the step's
`Primitive`; so far only `dot_general` has a kernel
([`../dot_general/tpu.rs`](../dot_general/README.md)), and any other step
errors (`<primitive> does not run on TPU yet`). The host-to-TPU copy
(`../device_transfer/tpu.rs`) is a kernel too.

Kernels are lumen's own, in **Mosaic**: the MLIR dialect a TPU core's
kernels are written in (Pallas lowers to it; XLA compiles it through a
`tpu_custom_call`). lumen writes the Mosaic module as text, wraps it in a
one-op StableHLO program, and libtpu compiles that through PJRT
(`lumen_tpu_compile`, `../../device/tpu/pjrt.c`). No XLA graph does the
work: the program is the kernel and nothing else.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `program` (the StableHLO wrapper), `base64`, the process-wide compile cache (`compiled`, keyed by the program's text), `Launch` / `launch`, and `encode` (per-primitive routing of plan steps). |
| `mosaic.rs` | `Kernel`, a builder of Mosaic modules in libtpu's serialized form, and its `Type` / `Space` / `Value`: arguments, scalars, buffer slices and reshapes, DMA copies, vector loads and stores, integer and float ops, `matmul`, `when` (`scf.if`), `finish`. |
| `tests.rs` | `base64`; on a TPU (one at a time, the allocator tests' lock): copies to and from the TPU beside their neighbours, float32 and bfloat16 dots against the reference, and a dot it cannot block failing with why. |

## How a kernel runs

A TPU's memory is the TPU allocator's one buffer, `u32[rows, 1, 128]`: rows
of 512 bytes, laid out byte for byte as the host does (`(1, 128)` tiles).
Every program is

```text
main(scalars: i32[n], [host: u32[h, 1, 128]], memory: u32[rows, 1, 128] {aliased to the result})
  -> u32[rows, 1, 128] = tpu_custom_call(scalars, [host], memory)   // the Mosaic kernel
```

`launch` passes the rows of the buffers the kernel reads and writes (TPU
pointers, each at the start of a row, as the allocator hands them out) as
its first scalars, uploads `host` bytes if any, and runs it with the memory
**donated**: the result is the same memory, whose new handle the allocator
keeps (`allocator::tpu::run`). Runs are asynchronous and ordered by that
chain of handles: each takes the previous one's result, and reads
(`copy_to_host`) wait for the latest.

A kernel (`mosaic::Kernel`) takes, in order: its grid's indices, the
scalars (SMEM), the host rows, the memory, the result (the same memory, as
written), its VMEM scratch and a DMA semaphore. Inside, it slices the
memory at a scalar's row, reshapes the rows to a matrix (a reshape of
`(1, 128)`-tiled memory, so its last dimension must be a multiple of 128
words), and copies blocks between it and VMEM by DMA.

## Invariants & gotchas

* **Text in libtpu's serialized form**: each op generic and named
  `stable_mosaic.<dialect>.<op>`, the module carrying
  `stable_mosaic.version` (`VERSION`, 17; libtpu reads it and older ones).
  Properties are written as attributes (`{...}`), as `mosaic-serde` writes
  them; `scf.if` has both regions (an empty `else`); the semaphore counts
  among `scratch_operands`; `window_params` has one entry per argument after
  the scalars, the result included.
* **Dynamic minor offsets** must be marked multiples of 128
  (`multiple_of`), or Mosaic cannot prove the DMA's alignment.
* **One program per text**: a kernel's text includes its shapes and the
  memory's row count, so each distinct shape compiles once per process
  (a host-to-TPU copy of each distinct row count, too).
* **Rows, not bytes**: a buffer a kernel touches must start at a row
  (`allocator::tpu::run` errors otherwise). The allocator rounds every
  allocation up to whole rows; a plan's workspace offsets are aligned to 256
  bytes only, which is fine while every TPU buffer is whole rows (a dot's
  operands and output are).
* **Portable**: programs are compiled for no device in particular and run on
  the device of the memory they are given (`execute_device`).

## Related

* [`../dot_general/`](../dot_general/README.md) — the dot kernel (`tpu.rs`).
* [`../device_transfer/`](../device_transfer/README.md) — host ↔ TPU copies.
* [`../../device/tpu/`](../../device/tpu/README.md) — the PJRT shim: compile and run.
* [`../../allocator/`](../../allocator/README.md) — `TpuAllocator`: the memory, its rows, `run`.
