# gather

The TPU's row gather: it picks rows of a `[vocab, width]` table at int32 indices along axis 0, for example an
embedding lookup. MPS and Triton have no gather kernel; they gather inside their elementwise fusions.

## Files

- `mod.rs` declares the `tpu` submodule, built only with `lumen_tpu_linked`. This family has no codegen trait.
- `tpu/mod.rs` encodes the gather:
  - `kernel(step)` builds the TensorCore kernel's `codegen::utils::tpu::Kernel` (module
    `lumen.codegen.gather.tpu.kernel`, `Rows::Step`). It returns an error for any gather the kernels don't
    cover; `checked` accepts only table rows at int32 indices along axis 0.
  - The spec's blocks are bounded by `BLOCK_BYTES`, `ROWS` and the index window `WINDOW`. Each output block
    holds whole TPU tiles of rows, and the index windows must tile the index array.
  - `sparse_core(device, step, inputs, output)` launches the gather on the SparseCore and returns `true` when
    the table's rows allow it: rows must be 32-bit, cover whole lanes and be tile-aligned in the shared
    region. Chunks are bounded by `SPARSE_CORE_CHUNK`. Otherwise it returns `false`.
- `tpu/kernel.py`: `program(spec)` builds one of two Pallas programs.
  - The SparseCore program (`_sparse_core`): each subcore gathers its share of the indices `k` rows at a time by
    indirect DMA. It then writes them to consecutive output rows, or to explicit destination rows when the
    output is not linear.
  - The TensorCore program: it DMAs the tile holding each wanted row, double-buffering the next block's
    fetches, and picks the row's sublane (and its half-word, for packed 16-bit rows) out of the tile.

## How it runs

- The SparseCore or TensorCore choice depends on the buffers' layout, which is known only at launch. For that
  reason `codegen/utils/tpu` gives every `Gather` step the `Gather` kernel (defined in `codegen/utils/tpu`, not here).
- At codegen, `codegen/utils/tpu` tries to build the TensorCore kernel. A gather that kernel rejects keeps no code
  rather than failing.
- When the step runs, `Gather` tries `sparse_core` first. If that declines, it runs the TensorCore kernel
  through `Pallas`. If neither can run the gather, it fails with the TensorCore kernel's error.
- Tables are stored TPU-tiled, so the kernels view them as `int32` words through `pallas.view` on the shared
  memory.
