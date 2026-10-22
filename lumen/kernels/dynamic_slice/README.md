# `lumen/kernels/dynamic_slice`

Kernels for the `dynamic_slice` and `dynamic_update_slice` primitives: a
block of an operand at start indices that are only known at run time (scalar
tensors, read on the device). MPS runs both (`mps/`, compiled when
`lumen_mps_linked` is set); TPU runs `dynamic_update_slice` (`tpu/`, when
`lumen_tpu_linked` is).

Callers: graph plans on MPS reach [`mps::encode`](mps/mod.rs) through the
primitive dispatch in [`lumen/kernels/mps/mod.rs`](../mps/mod.rs), on TPU
[`tpu::encode`](tpu/mod.rs) through [`lumen/kernels/tpu/mod.rs`](../tpu/mod.rs). The reference
interpreter (`lumen/kernels/reference.rs`) has its own `DynamicSlice` /
`DynamicUpdateSlice` arms and does not use this folder. The plain device copy
`mps::copy` is reused elsewhere as lumen's MPS buffer-to-buffer copy (see
[`mps/README.md`](mps/README.md)).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `#[cfg(lumen_mps_linked)] mod mps;` and `#[cfg(lumen_tpu_linked)] mod tpu;` hooks. |
| `tpu/mod.rs` | `encode`: checks the update is one the kernel runs (rank 2 or more, 16- or 32-bit elements, int32 starts), picks its blocks and whether the operand is copied first, and writes the spec. |
| `tpu/__init__.py` | `program(spec)`: the TPU kernel (below). |

## On TPU

A step is one `dynamic_update_slice` (starts read on the device: DMA'd into
SMEM and clamped as XLA's) or a `dynamic_update_slices` fusion: a chain of
updates of one shape at constant starts, each into the last's result
(`graph_compiler/tpu/updates.rs`), whose starts `encode` clamps and passes
as the launch's scalars. The operand is updated in place when the plan gives
the output its memory; otherwise it is first copied into the output through
VMEM (one HBM-to-HBM DMA streams at ~64 GB/s, so nothing moves HBM to HBM).

Mosaic DMAs and stores only at offsets it can prove tile-aligned (an
unaligned DMA misplaces data silently), so an update that is not
(rows and columns of whole tiles, starting on a tile) is rolled into place
(`pl.roll`) and merged under a mask with the tile-aligned window around it
(16-bit values widened to 32 bits). Kernels by what `encode` knows:

* **Resident** (constant starts, the chain's tile-aligned box within 48 MiB
  of VMEM, updates every one aligned, or overlapping ones of one block):
  the box is read into VMEM once, unless the updates tile it exactly (and
  then a copy of the operand skips its rows); each update is DMA'd into it
  (all at once if no two overlap) or merged into it in order, and the box
  is written once.
* **One block each**: updates are prefetched 8 ahead; an aligned one is
  written from VMEM, any other merged with its window. If no two updates'
  windows overlap, each one's write overlaps the next.
* **Larger**: per update, aligned ones are staged through VMEM in blocks of
  up to 1M elements, double-buffered; any other streams in blocks of whole
  rows: each output block is the update's block and the last tile of the one
  before, rolled into place; only blocks on its first or last row read the
  output whole, the others its two edge strips of 256 columns.

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder. [`tpu/`](tpu/) — the TPU kernel and its encoder.

## Related

- [`lumen/kernels/mps`](../mps/) — `launch`, `Grid`, arg helpers, shared Metal prelude.
- [`lumen/kernels/scatter_add`](../scatter_add/) — `scatter_add` reuses `mps::copy`.
- `lumen/kernels/reference.rs` — the CPU reference semantics.
