# `lumen/kernels/gather`

The `gather` templates on TPU: an embedding's lookup, `table[ids]`, the rows
of a `[vocab, width]` table at int32 indices of any shape. A 32-bit table
(float32, int32) is gathered on the TPU's **SparseCores**; a bfloat16 one,
or a 32-bit one the SparseCore path cannot take, on the TensorCore. On MPS a gather
has no kernel: it is generated (in a fusion, or alone as a one-node fusion,
`graph_compiler/mps/alone.rs`).

Callers: plan steps on a TPU reach [`tpu::encode`](tpu/mod.rs) through
`lumen/kernels/tpu/mod.rs`'s `encode` (`Gather { .. }`).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `#[cfg(lumen_tpu_linked)] mod tpu;` hook. |
| `tpu/mod.rs` | `encode`: checks the gather is one the templates run, launches the SparseCore one where it can (`sparse_core`), else picks the TensorCore one's blocks and writes its spec. |
| `tpu/__init__.py` | `program(spec)`: the kernels (`_sparse_core` for the SparseCores). |

## On the SparseCores (32-bit tables)

`tpu::encode` launches a SparseCore kernel (`_sparse_core`) when the table
is 32-bit, its width a multiple of 128 with at least 8 rows in 128 KiB, and
the table, the output and the memory start on tile rows of that width (the
allocator's and the plan's alignment makes them). The launch
(`lumen/kernels/tpu`'s SparseCore launches) gives the kernel the memory as
`u32[_, width]` rows and each position's table row (its id plus the
table's row). Each of the 32 subcores takes its share of positions in
chunks of `k` rows (the most a power of two that fit 128 KiB: the chunk
size, not more chunks in flight, is what makes it fast), copies the
chunk's rows in by one indexed copy (a hardware gather, 32 bytes at a
time: no tile is over-read) and writes them out:

* **linearly**, where the output's rows are contiguous (1-D ids, or a last
  index dimension a multiple of 8): the launch passes its row scalars, the
  kernel reads the output's row from them (HBM to shared VMEM to SMEM: a
  vector subcore copies nothing from HBM to SMEM itself) and writes each
  chunk at its position, the last part chunk in whole tiles (its tail in
  the output's tile padding);
* else by an indexed copy at each position's output row (a second index
  array).

On a v6e, a lookup takes as long as JAX's own SparseCore gather (Pallas,
`tpu_sc`) on the same shapes, 2.5x faster than XLA's `jnp.take`: per lookup
in a program of several, `[8, 2048]` ids in a `[32000, 1024]` float32 table
139 µs (484 GB/s written; JAX 138, XLA 334), `[8, 4096]` in `[32000, 256]`
72 µs (469 GB/s; JAX 70, XLA 107), `[4, 1024]` in `[8000, 4096]` 139 µs
(JAX 139, XLA 226).

## On the TensorCore

A TPU copies whole tiles: a table row alone cannot be copied (a copy of one
row starts at its tile's first). So for each index the kernel copies the
tile holding its row (8 rows of 32-bit words) into VMEM and picks the row
out with vector ops: each output row is the sum, in int32, of its tile's
rows masked to the one it wants, so the bits are the table's exactly (NaNs,
signed zeros, subnormals).

* **Views**: the table, the indices and the output are viewed as TPU lays
  them out: indices `[s]`, or `[lead, r, s]` slabs (`[..., r, s]`), the
  output `[lead * r, s, width]`. A 16-bit table is viewed as int32 rows of
  two packed rows each (the TPU's packing: row `2w` in word `w`'s low half);
  each index's word is shifted to its half, two rows packed into each output
  word, so no 16-bit value is ever converted.
* **Blocks**: `bm` rows (up to 64, whole tiles) of `bn` columns (as wide as
  8 MiB of tiles allows) over the grid `[lead * r, s / bm, width / bn]`.
* **Indices**: each block's from a `[rw, cw]` window of the indices (whole
  tiles: 8 rows by 128, or all of them when they fit `WINDOW` words) copied
  to SMEM.
* **Pipelined**: the next block's window and tiles are fetched while this
  one's rows are picked (two slots), the output written by `emit_pipeline`.

Throughput is bound by the tiles read: 8 times the rows a block writes (16
for a 16-bit table). On a v6e, a lookup of `[8, 2048]` ids in a
`[32000, 4096]` bfloat16 table reads tiles at ~1 TB/s (64 GB/s written).

## Not yet

A gather along other than the first axis, of a table of other than two
dimensions, or with indices that are not int32 (`int64` is not stored on a
TPU) does not run on TPU yet; nor do `take_along_axis` and `scatter_add`
(a gather's gradient).
