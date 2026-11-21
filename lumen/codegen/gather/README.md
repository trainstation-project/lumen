# gather

The TPU's gathers: the row gather picks rows of a `[vocab, width]` table at int32 indices along axis 0, for example an
embedding lookup, and `take_along_axis` reads each element of an operand at its index along an axis. MPS and Triton
have no gather kernel; they gather inside their elementwise fusions (`take_along_axis` lowered to a flat gather).

## Files

- `mod.rs` declares the `tpu` submodule, built only with `lumen_tpu_linked`. This family has no codegen trait.
- `tpu/mod.rs` encodes the gather:
  - `kernel(step)` builds the TensorCore kernel's `codegen::devices::tpu::Kernel` (module
    `lumen.codegen.gather.tpu.kernel`, `Rows::Step`). It returns an error for any gather the kernels don't
    cover; `checked` accepts only table rows at int32 indices along axis 0.
  - The spec's blocks are bounded by `BLOCK_BYTES`, `ROWS` and the index window `WINDOW`. Each output block
    holds whole TPU tiles of rows, and the index windows must tile the index array.
  - `sparse_core(device, step, inputs, output)` launches the gather on the SparseCore and returns `true` when
    the table's rows allow it: rows must be 32-bit, cover whole lanes and be tile-aligned in the shared
    region. Chunks are bounded by `SPARSE_CORE_CHUNK`. Otherwise it returns `false`.
  - `take_along_axis(step)` builds the `take_along_axis` kernel's spec: a 32-bit operand of rank 2 or more at int32
    indices, each viewed `[slabs, m, n]` at its tile-padded shape (`Layout::padded`), so it is never flattened; the
    kind by the axis (`lanes`, `sublanes` or `outer`) and the operand's slab strides. Others are rejected.
- `tpu/take_along_axis.py`: `program(spec)` fills `[8, 128]` output blocks. Along the last axis a block reads its
  rows of the operand whole and gathers them a register of 128 columns at a time (Mosaic gathers within one
  register), keeping each element from the register its index falls in; along the second-last axis the same with its
  columns, a register of 8 rows at a time; along an earlier axis it fetches the operand's block at each position and
  keeps it where the index is that position.
- `tpu/kernel.py`: `program(spec)` builds one of two Pallas programs.
  - The SparseCore program (`_sparse_core`): each subcore gathers its share of the indices `k` rows at a time by
    indirect DMA. It then writes them to consecutive output rows, or to explicit destination rows when the
    output is not linear.
  - The TensorCore program: it DMAs the tile holding each wanted row, double-buffering the next block's
    fetches, and picks the row's sublane (and its half-word, for packed 16-bit rows) out of the tile.

## How it runs

- The SparseCore or TensorCore choice depends on the buffers' layout, which is known only at launch. For that
  reason `codegen/devices/tpu` gives every `Gather` step the `Gather` kernel (in `tpu/mod.rs`).
- At codegen, `codegen/devices/tpu` tries to build the TensorCore kernel. A gather that kernel rejects keeps no code
  rather than failing.
- When the step runs, `Gather` tries `sparse_core` first. If that declines, it runs the TensorCore kernel
  through `Pallas`. If neither can run the gather, it fails with the TensorCore kernel's error.
- Tables are stored TPU-tiled, so the kernels view them as `int32` words through `pallas.view` on the shared
  memory.
