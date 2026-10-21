# `lumen/kernels/gather`

The `gather` template on TPU: an embedding's lookup, `table[ids]`, the rows
of a `[vocab, width]` table at int32 indices of any shape. On MPS a gather
has no kernel: it is generated (in a fusion, or alone as a one-node fusion,
`graph_compiler/mps/alone.rs`).

Callers: plan steps on a TPU reach [`tpu::encode`](tpu/mod.rs) through
`lumen/kernels/tpu/mod.rs`'s `encode` (`Gather { .. }`).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `#[cfg(lumen_tpu_linked)] mod tpu;` hook. |
| `tpu/mod.rs` | `encode`: checks the gather is one the template runs, picks the blocks and writes the spec. |
| `tpu/__init__.py` | `program(spec)`: the kernel. |

## How it runs

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
