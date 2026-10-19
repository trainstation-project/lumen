# `lumen/ops/tpu`

TPU kernels and how they run. A plan step on a TPU (`Plan::execute`, the
executor `Device::Tpu`) goes to `encode`, which routes on the step's
`Primitive`; so far only `dot_general` has a kernel
([`../dot_general/`](../dot_general/README.md)), and any other step errors
(`<primitive> does not run on TPU yet`). The host-to-TPU copy
(`../device_transfer/tpu.rs`) is a kernel too.

Kernels are lumen's own, written in **lumen's Pallas**
([`../../pallas/`](../../pallas/README.md): Pallas's API, traced and lowered
to Mosaic by lumen, no JAX), as MPS kernels are Metal templates: a kernel
is a Python module beside its op (`lumen/ops/<op>/tpu.py`) whose
`program(spec)` builds the kernel for a spec that Rust chose and lowers it.
**Rust decides, the kernel computes**: the Rust side of an op
(`lumen/ops/<op>/tpu.rs`) checks what the kernel can run, picks blocks and
writes the spec; the kernel's Python traces it and lumen lowers it to
Mosaic; Rust wraps the Mosaic in a StableHLO program of one
`tpu_custom_call`, libtpu compiles it (`lumen_tpu_compile`), and lumen runs
it on the device's memory. No XLA graph does the work: the program is the
kernel and nothing else.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `lower` (calls `pallas.lower` through PyO3; without the `python` feature, an error), `program` (the StableHLO wrapper) and `base64`, the process-wide cache of compiled programs (`compiled`, by module, spec and arguments), `Launch` / `launch` (adds the scalars' count, host rows and memory rows to the spec, runs the program), and `encode` (per-primitive routing of plan steps). |
| `pallas.py` | `lower(module, spec)` (Rust's entry: `module.program(spec)`), `program` (a kernel traced on the device's memory of a spec, `lumen.pallas.kernel`), `matrix` (a matrix of words at a row of the memory). |

## How a kernel runs

A TPU's memory is the TPU allocator's one buffer, `u32[rows, 1, 128]`: rows
of 512 bytes, laid out byte for byte as the host does (`(1, 128)` tiles).
Every program (`mod.rs`'s `program`) is

```text
main(scalars: i32[n], [host: u32[h, 1, 128]], memory: u32[rows, 1, 128] {aliased to the result})
  -> u32[rows, 1, 128] = tpu_custom_call(scalars, [host], memory)   // the kernel
```

and its kernel is `kernel(scalars, [host,] memory, result, *scratch)`: the
scalars in SMEM, the memory in HBM, read as `memory` and written as
`result` (the same buffer). `pallas.matrix(memory, row, shape)` is a matrix
of 32-bit words at a row (a window of rows, reshaped: its last dimension a
multiple of 128 words), which a kernel can pipeline over
(`pl.emit_pipeline`, its blocks' copies double-buffered against compute).

`launch` passes the rows of the buffers the kernel reads and writes (TPU
pointers, each at the start of a row, as the allocator hands them out) as
its first scalars, uploads `host` bytes if any, and runs the program with
the memory **donated**: the result is the same memory, whose new handle the
allocator keeps (`allocator::tpu::run`). Runs are asynchronous and ordered
by that chain of handles: each takes the previous one's result, and reads
(`copy_to_host`) wait for the latest.

With `LUMEN_TPU_DUMP=dir`, each program is written there as it is compiled
(`<pid>.NNNN.<op>.mlir`: the StableHLO, after a comment line of the module and
spec), as XLA's `--xla_dump_to` does: to run or profile elsewhere (JAX's
backend compiles it as it is).

## Invariants & gotchas

* **Python only.** Kernels are traced from Python: TPU kernels run where
  lumen runs from Python (the `python` feature). The Rust tests of a TPU
  stop at its memory (`allocator`); kernels' lowering is tested in Rust
  (`lumen/pallas/tests.rs`), their runs from Python (`tests/test_tpu.py`).
* **Specs are numbers.** A spec carries what the kernel needs as numbers
  (packing: elements a 32-bit word, not dtype names), its fields JSON that
  Rust writes by hand.
* **One program per spec**: each distinct shape traces and compiles once
  per process (a host-to-TPU copy of each distinct row count, too).
* **Rows, not bytes**: a buffer a kernel touches must start at a row
  (`allocator::tpu::run` errors otherwise). The allocator rounds every
  allocation up to whole rows; a plan's workspace offsets are aligned to 256
  bytes only, which is fine while every TPU buffer is whole rows (a dot's
  operands and output are).
* **Mosaic version 17**, which the libtpu of the `tpu` extra reads (0.0.48
  on).
* **Portable**: programs are compiled for no device in particular and run on
  the device of the memory they are given (`execute_device`).

## Related

* [`../../pallas/`](../../pallas/README.md) — lumen's Pallas: the kernel API, its IR and Mosaic lowering.
* [`../dot_general/`](../dot_general/README.md) — the dot (`tpu.rs`: its spec; `tpu.py`: its kernel).
* [`../device_transfer/`](../device_transfer/README.md) — host ↔ TPU copies.
* [`../../device/tpu/`](../../device/tpu/README.md) — the PJRT shim: compile and run.
* [`../../allocator/`](../../allocator/README.md) — `TpuAllocator`: the memory, its rows, `run`.
