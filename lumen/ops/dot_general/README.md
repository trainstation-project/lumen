# `lumen/ops/dot_general`

Kernels for the `dot_general` primitive (XLA's batched, generalized
matmul) and for `linear_cross_entropy` (the logits `h · wᵀ` fused with a
cross-entropy, never materialized). There is no `Op` here: `dot_general` is
a graph primitive, run by a compiled plan. On MPS, `ops::mps::encode` sends
its steps to `mps::encode` / `mps::encode_linear_cross_entropy`; on a TPU,
`ops::tpu::encode` sends `dot_general` to `tpu::encode` (its kernel, in lumen's Pallas:
`tpu.py`); elsewhere the plan
evaluates it with `ops/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` and `#[cfg(lumen_tpu_linked)] mod tpu;` hooks. |
| `tpu.rs` | `dot_general` on TPU: its form (`form`: batch, then `[M, K] · [K, N]` or `[N, K]`), block choice (`block`), and its kernel's spec; see *TPU* below. |
| `tpu.py` | The kernel of a dot on TPU, in lumen's Pallas (`program(spec, epilogue)`). |

## TPU

`tpu.py` is the kernel, in lumen's Pallas ([`../../pallas/`](../../pallas/README.md)); `tpu.rs` its spec
([`../tpu/`](../tpu/README.md) lowers and runs it). The lhs must be batch,
free then contracting dimensions; the rhs batch, then contracting and free
dimensions (`[K, N]`) or free and contracting (`[N, K]`, `transposed`);
anything else errors. Operands are float32 or bfloat16 (one dtype),
accumulating in float32; the output float32 or bfloat16.

* **Pipeline** over `(batch, M/bm, N/bn, K/bk)` (`pl.emit_pipeline`, the
  blocks' copies double-buffered against compute): each step multiplies an
  lhs and rhs block on the matrix unit into a float32 accumulator (VMEM,
  zeroed at the first k step) and the last k step writes the output
  block, through the kernel's `epilogue` (a float32 block's function, not
  used yet), in the output's dtype. One block of k needs no accumulator: the
  product is written as it is. One product a step: Mosaic unrolls it over
  the block's matrix-unit tiles, so a second copy of it costs dearly.
* **Blocks** (`tpu.rs`, `caps`): as large as feed the matrix unit (a step
  does `bm · bn / (bm + bn)` flops a byte read; a v6e's HBM moves a byte
  while it does ~800), measured on a v6e (`benchmarks/tpu_matmul.py`):
  bfloat16 2048 × 1024 × K for K ≤ 4096 (no accumulator), else
  2048 × 2048 × 2048; float32 (full precision, six bfloat16 passes)
  2048 × 1024 × 1024. Each is the whole dimension if smaller, else the
  largest multiple of a tile's rows (M) or of 128 (N, K) dividing it,
  shrunk (k's first) until its VMEM (`footprint`: double buffers, the
  accumulator, float32's bfloat16 parts) fits 88 MiB of the kernel's
  100 MiB; then n's halved (to 256) until the grid has 4 steps, so small
  dots' copies overlap their products. On a v6e (one core pinned, a second
  a shape) large dots run at ~770–785 TFLOP/s in bfloat16 (XLA's
  730–790) and ~132 in float32 (XLA's 124): XLA's or faster at every shape
  of `benchmarks/tpu_matmul.py` but bfloat16 4096³ (98.6%: its first
  blocks' 24 MB copy, which nothing overlaps).
  Shapes that cannot block error, naming the dimension.
* **Tiles.** A TPU keeps tensors in its own tiles (`device/tpu/layout.rs`),
  as Mosaic reads them: the kernel views its operands and output in place
  (`tpu.reinterpret_cast`), with no packing or reshuffling of words.
* float32 operands multiply in full float32 (`contract_precision<fp32>`),
  not the matrix unit's single bfloat16 pass.

## Subfolders

* [`mps/`](mps/README.md) — the Metal matmul kernels (integer, 64-bit
  integer, simdgroup-matrix float, split-K atomic, linear-cross-entropy) and
  the host code that canonicalizes a `dot_general` to `[b, m, k] × [b, k, n]`
  and picks a kernel and tile.

## Related

* `lumen/graph_compiler/mps/canonicalize_dots.rs`, `split_k.rs`,
  `place_views.rs` — compiler passes that use this folder's matmul-form
  helpers.
* `lumen/codegen/mps/mod.rs` — fusion kernels that call the `matmul_sg_impl` template.
