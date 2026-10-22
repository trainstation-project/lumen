# `lumen/kernels/tpu`

TPU kernels and how they run. A plan step on a TPU (`Plan::execute`, the
executor `Device::Tpu`) goes to `encode`, which routes on the step's
`Primitive`; so far `dot_general` ([`../dot_general/`](../dot_general/README.md)),
loop, row and reduction fusions (generated: [`../../codegen/tpu/`](../../codegen/tpu/README.md);
reductions on the template in [`../reduce/tpu/`](../reduce/tpu/)) and
`gather` ([`../gather/`](../gather/README.md)) and `dynamic_update_slice`
([`../dynamic_slice/`](../dynamic_slice/README.md)) have kernels, and any other step errors
(`<primitive> does not run on TPU yet`). The host-to-TPU copy
(`../device_transfer/tpu/mod.rs`) is a kernel too.

Kernels are lumen's own, written in **lumen's Pallas**
([`../../pallas/`](../../pallas/README.md): Pallas's API, traced and lowered
to Mosaic by lumen, no JAX), as MPS kernels are Metal templates: a kernel
is a Python module beside its op (`lumen/kernels/<op>/tpu.py`) whose
`program(spec)` builds the kernel for a spec that Rust chose and lowers it.
**Rust decides, the kernel computes**: the Rust side of an op
(`lumen/kernels/<op>/tpu.rs`) checks what the kernel can run, picks blocks and
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
Every program (`mod.rs`'s `program`) is a chain of kernels, each a
`tpu_custom_call` on its slice of the scalars and the memory the one before
returned:

```text
main(scalars: i32[n], [host: u32[h, 1, 128]], memory: u32[rows, 1, 128] {aliased to the result})
  -> u32[rows, 1, 128]
  m0 = tpu_custom_call(scalars[0:n0], [host], memory)   // the first kernel
  m1 = tpu_custom_call(scalars[n0:n0+n1], [host], m0)   // the next, ...
  return mK
```

and each kernel is `kernel(scalars, [host,] memory, result, *scratch)`: its
scalars in SMEM, the memory in HBM, read as `memory` and written as
`result` (the same buffer). `pallas.matrix(memory, row, shape)` is a matrix
of 32-bit words at a row (a window of rows, reshaped: its last dimension a
multiple of 128 words), which a kernel can pipeline over
(`pl.emit_pipeline`, its blocks' copies double-buffered against compute).

`launch` lowers a kernel (once per spec, `lowered`) and gives it the rows
of the buffers it reads and writes (TPU pointers, each at the start of a
row, as the allocator hands them out) as its first scalars. Inside a
**`Batch`** (a plan's run on a TPU: `Plan::execute`) launches are queued,
and the batch runs them as **one program**, as XLA runs a jitted function's
Pallas kernels: launched once, with no program switches between its kernels
(a TPU loads a program's code when it switches to it: a large dot's
unrolled blocks, ~2 MiB, take ~14 µs). A program is compiled once per
sequence of kernels (`compiled`). A launch with `host` bytes (a host-to-TPU
copy) runs on its own, after the batch's launches so far; one outside a
batch runs at once. A program runs with the memory **donated**: the result
is the same memory, whose new handle the allocator keeps
(`allocator::tpu::run`). Runs are asynchronous and ordered by that chain of
handles: each takes the previous one's result, and reads (`copy_to_host`)
wait for the latest.

## SparseCore launches

A `Launch` with `sparse: Some(SparseCore)` runs on a TPU's SparseCores
(`SPARSE_CORE_MESH`), in the same program as the TensorCore kernels.
SparseCore kernels see no scalars and no views of the memory, so `program`
wraps the custom call in HLO that gives them both:

* **Index arrays** (`Indices`): int32 values at a buffer's row (a dynamic
  slice of the memory at its row scalar, a tiled `[.., r, s]` tensor
  relaid out row-major) or a strided iota, plus a buffer's row in units of
  the view's rows; padded to `count` with their last value.
* **The view**: the memory `u32[rows, 1, 128]` as `u32[rows * 128 / width,
  width]` (reshape, transpose of each tile's rows and columns, reshape),
  which XLA compiles to bitcasts: a tiled `[n, width]` 32-bit tensor
  starting on a tile row of the view (a multiple of `8 * width / 128`
  rows) is rows of it. A `bitcast_convert` of the whole memory is not free
  (XLA copies it): the view stays `u32`.
* **The call**: `DEVICE_TYPE_SPARSECORE` with `sparse_core_config` (TensorCore
  tiling, `TILING_COMPACT`), the view aliased to its result, then the
  inverse view back to the memory.

## Invariants & gotchas

* **Python only.** Kernels are traced from Python: TPU kernels run where
  lumen runs from Python (the `python` feature). The Rust tests of a TPU
  stop at its memory (`allocator`); kernels' lowering is tested in Rust
  (`lumen/pallas/tests.rs`), their runs from Python (`tests/test_tpu_kernels.py`).
* **Specs are numbers.** A spec carries what the kernel needs as numbers
  (packing: elements a 32-bit word, not dtype names), its fields JSON that
  Rust writes by hand.
* **One program per plan**: each distinct kernel spec traces once per
  process, and each plan's sequence of kernels compiles once (a host-to-TPU
  copy of each distinct row count, too).
* **Rows, not bytes**: a buffer a kernel touches must start at a row
  (`allocator::tpu::run` errors otherwise). The allocator rounds every
  allocation up to whole rows, and a tiled tensor up to 40 tiles (5120
  32-bit columns) wide starts on a tile row of its own
  (`device::tpu::layout::alignment`: 4 KiB a 128-column tile, less than its
  padded size), in the allocator and in a plan's workspace
  (`DeviceLayout::align`), which SparseCore launches rely on.
* **Mosaic version 17**, which the libtpu of the `tpu` extra reads (0.0.48
  on).
* **Portable**: programs are compiled for no device in particular and run on
  the device of the memory they are given (`execute_device`).

## Related

* [`../../pallas/`](../../pallas/README.md) — lumen's Pallas: the kernel API, its IR and Mosaic lowering.
* [`../dot_general/`](../dot_general/README.md) — the dot (`tpu.rs`: its spec; `tpu.py`: its kernel).
* [`../../codegen/tpu/`](../../codegen/tpu/README.md) — generated loop fusions of elementwise primitives (`mod.rs`: their spec; `__init__.py`: their program).
* [`../reduce/tpu/`](../reduce/tpu/) — the reduction template: rows, columns or total, in blocks of an accumulator (`mod.rs`: its blocks; `__init__.py`: its kernel).
* [`../device_transfer/`](../device_transfer/README.md) — host ↔ TPU copies.
* [`../../device/tpu/`](../../device/tpu/README.md) — the PJRT shim: compile and run.
* [`../../allocator/`](../../allocator/README.md) — `TpuAllocator`: the memory, its rows, `run`.
