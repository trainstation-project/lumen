# `lumen/kernels/dot_general`

Kernels for the `dot_general` primitive (XLA's batched, generalized
matmul) and for `linear_cross_entropy` (the logits `h · wᵀ` fused with a
cross-entropy, never materialized). There is no `Op` here: `dot_general` is
a graph primitive, run by a compiled plan. On MPS, `kernels::mps::encode` sends
its steps to `mps::encode` / `mps::encode_linear_cross_entropy`; on a TPU,
`kernels::tpu::encode` sends `dot_general` to `tpu::encode` (its kernel, in lumen's Pallas:
`tpu/__init__.py`); elsewhere the plan
evaluates it with `kernels/reference.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | Module doc and the `#[cfg(lumen_mps_linked)] mod mps;` and `#[cfg(lumen_tpu_linked)] mod tpu;` hooks. |
| `tpu/mod.rs` | `dot_general` on TPU: its form (`form`: batch, then `[M, K] · [K, N]` or `[N, K]`), block choice (`block`), and its kernel's spec, its epilogue's from `codegen/tpu/gemm.rs`; see *TPU* below. |
| `tpu/__init__.py` | The kernel of a dot on TPU, in lumen's Pallas (`program(spec, epilogue)`). |
| `triton.py` | The Triton backend's GEMM template (`use_triton_backend`), from open-lm-engine/accelerated-model-architectures's `bmm_triton`: `gemm_kernel`, `L` batches of `[M, K] · [K, N]`, tiled by one of 7 configurations (`codegen/triton/config.rs`'s `CONFIGS`, which `graph_compiler/triton`'s `Gemms` tunes per dot with lumen's CUDA events when `autotune` is on), `GROUP_SIZE_M` row-blocks at a time so a group's tiles reuse the same columns of B from cache, accumulated in `ACCUM_DTYPE` (exactly: no TF32) with int64 indices. It knows no layout: it loads each tile through `PROLOGUE_A` / `PROLOGUE_B` and stores through `EPILOGUE`, generated for the operands' and output's dimensions and strides (`codegen/triton/gemm.rs`). `dot_general(tensors, sizes, accum_dtype, config, functions)` launches it. |

## TPU

`tpu/__init__.py` is the kernel, in lumen's Pallas ([`../../pallas/`](../../pallas/README.md)); `tpu/mod.rs` its spec
([`../tpu/`](../tpu/README.md) lowers and runs it). The lhs must be batch,
free then contracting dimensions; the rhs batch, then contracting and free
dimensions (`[K, N]`) or free and contracting (`[N, K]`, `transposed`);
the lhs may also be batch, contracting then free (`[K, M]`,
`lhs_transposed`: a weight's gradient `dyᵀ x`, multiplied by Mosaic's
dot_general contracting the lhs's first dimension); anything else errors. Operands are float32 or bfloat16 (one dtype),
accumulating in float32; the output float32 or bfloat16.

* **Pipeline** over `(batch, M/bm, N/bn, K/bk)` (`pl.emit_pipeline`, the
  blocks' copies double-buffered against compute): each step multiplies an
  lhs and rhs block on the matrix unit into a float32 accumulator (VMEM,
  zeroed at the first k step) and the last k step writes the output
  block, rounded to the dot's dtype, then through its epilogue (below). One block of k needs no accumulator: the
  product is written as it is. One product a step: Mosaic unrolls it over
  the block's matrix-unit tiles, so a second copy of it costs dearly.
* **Blocks** (`tpu/mod.rs`, `caps`): as large as feed the matrix unit (a step
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
* **Fusions** (`fuse_dots` in `graph_compiler/tpu/fusion/dot_epilogues.rs`): a dot, its
  slices if merged (`merge_dots`: members of its columns; a merged weight
  gradient's: of its rows, `split_rows`), and the elementwise nodes after
  it that read one member's values (and otherwise constants, 0-d values or
  values of the member's shape computed before the dot) are one `Fusion`.
  A concatenate its lhs is, read only by dots, that no kernel writes in
  place, is the fusion's too: its parts are buffers of their own the
  kernel reads side by side along k or m (`lhs`, `lhs_k`). `codegen/tpu/gemm.rs`
  (`Epilogue::of`) writes the spec: `widths` (the members), `inputs` (the
  other operands: 0-d ones copied once, the rest read a block at a time at
  the member's blocks), `nodes` (as the elementwise kernel's, computed by
  its `computed`) and `outputs` (a value, its member, and the buffer it is
  written into: its own, or columns of a wider one, its step's output
  view). Blocks divide every member, part and column offset; a member's
  or part's block index is the grid's less its first block, clamped to its
  blocks, and only steps within it read or write it, so the pipeline
  copies each of its blocks once.
* **In-place concatenates** (`place_concats` in `graph_compiler/tpu`): a
  concatenate along the last dimension of values only it reads, each
  written by a dot's fusion or an elementwise kernel, has no step: each
  kernel writes at its columns of the concatenate's buffer. A merged
  backward's cotangents (`merge_dots_backward`) are so one buffer, which
  x's gradient's dot and the weights' gradients' dot read whole.
* On a v6e (bfloat16): relu of a 4096³ dot 185 µs fused, 281 µs as two
  kernels; a residual add 194 µs (256); `y * sigmoid(1.702 y)` in float32
  198 µs (724); q/k/v projections (`x [4096, 2048]`, 2048 + 512 + 512
  outputs) 136 µs merged, 161 µs as three dots; their backward (three
  2048 × 2048 weights, `x [4096, 2048]`) 323 µs in two dots, 380 µs in
  six. Reading the cotangents as three buffers instead of one costs the
  weights' gradients' dot (303 µs, 158 µs from one buffer): their three
  lhs streams' VMEM shrinks its blocks.

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
