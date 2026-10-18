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
  zeroed at the first K step), and the last K step writes the output
  block, through the kernel's `epilogue` (a float32 block's function, not
  yet passed by Rust) if any.
* **Blocks** (`tpu.rs`): `bm` is M if unbatched and at most the cap, else a
  multiple of 8 dividing it; `bn` and `bk` the largest multiples of their
  unit (128 words of a row) up to the cap dividing N and K. Caps, tuned on
  a v6e: 1024 × 1024 × 512 for bfloat16, 512 × 512 × 512 for float32.
  Shapes that cannot block error, naming the dimension.
* **Words.** The memory holds tensors row-major, as the host does; the
  kernel reads 32-bit words. A bfloat16 word holds two neighbours of a row,
  so a bfloat16 dot splits operands by parity: `out[:, q::2] = Σ_p
  lhs[:, p::2] · rhs[p::2, q::2]` (a rhs `[K, N]` pairs its rows by k
  parity: two rows a row of `[K/2, N]` words; a `[N, K]` one by n parity),
  each half widened to float32 by a shift and narrowed to bfloat16 exactly.
  A bfloat16 output is written as words: even columns' rounded bits in the
  low halves, odd columns' in the high (its accumulator holds the even
  columns, then the odd).
* **Supported**: float32 → float32; bfloat16 → bfloat16; bfloat16 → float32
  with a `[N, K]` rhs. The others (float32 → bfloat16, bfloat16 → float32
  with a `[K, N]` rhs) would interleave columns across lanes: not yet.
* float32 operands multiply in full float32 (`contract_precision<fp32>`),
  not the matrix unit's bfloat16 passes.
* On a v6e at 4096³, through `lumen.compile`: about 465 TFLOP/s in bfloat16
  (XLA's: 770) and 110 in float32 (125): JAX's Pallas kernel's speed.

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
