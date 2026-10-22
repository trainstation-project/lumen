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

The operand is updated in place when the plan gives the output its memory;
otherwise it is first copied into the output (a pipelined copy through VMEM:
one HBM-to-HBM DMA streams at ~64 GB/s). The starts are DMA'd into SMEM and
clamped as XLA's. For each of the update's leading slabs:

* **Aligned** (rows and columns of whole tiles, starting on a tile, checked
  at run time): the update's blocks are staged through VMEM, double-buffered,
  into the output.
* **Otherwise**: each block of the update reads the output's tile-aligned
  window around it, rolls the update into place (`pl.roll`), merges it under
  a mask (16-bit values widened to 32 bits) and writes the window back:
  Mosaic stores only at offsets it can prove tile-aligned.

When the update is one block it is fetched before the starts. Blocks are of
whole tiles (`bm` rows, `bn` columns, up to 512K elements).

## Subfolders

[`mps/`](mps/) — the Metal kernels and their host-side encoder. [`tpu/`](tpu/) — the TPU kernel and its encoder.

## Related

- [`lumen/kernels/mps`](../mps/) — `launch`, `Grid`, arg helpers, shared Metal prelude.
- [`lumen/kernels/scatter_add`](../scatter_add/) — `scatter_add` reuses `mps::copy`.
- `lumen/kernels/reference.rs` — the CPU reference semantics.
