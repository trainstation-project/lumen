# dot_general

GEMM code generation and kernels. A plan step that is a lone `dot_general`, a GEMM fusion (a dot with the
elementwise epilogue after it), a dot fused with a normalization row, or a linear cross entropy runs on one of
the kernels here. Backend codegen (`codegen/{cuda,triton,mps,tpu}.rs`) picks this family's `Kernel` for those
steps, and the autotuner times its configs.

## Files

- `mod.rs` — the `GemmCodegen` trait: `gemm(body, dot, by_value, options)` returns a backend's `Code` for a GEMM
  fusion, with backend-specific `Options` (MPS tile configs, Triton operand/output layouts, TPU's step).
  Also `RowTile`: the widest row one GEMM tile holds whole (`ROW_TILE`), which MPS and Triton implement and
  `graph_compiler::passes::linear_reordering` reads. It also defines the graph compiler's dot pass traits each backend implements in its device folder: `StridedOperands`, `MergesDots`, `LinearCrossEntropies`, `InnerProducts`, `FoldsDotReshapes`, `DotLayouts`, `SplitsK`, `PartialRows`, `FusesDots` and `RowTile`.
- `cuda/` — CUDA (CuTe DSL) GEMM.
- `triton/` — Triton GEMM, row-epilogue GEMMs and linear cross entropy.
- `mps/` — Metal GEMMs (tiled, MPP, skinny, GEMV, integer), row matmuls and linear cross entropy.
- `tpu/` — Pallas GEMM and GEMM-plus-row kernels.

## cuda

- `mod.rs`: `matmul` says which dots the GEMM runs (f16/bf16 operands of rank 2, or 3 with a leading batch dim,
  one contracting dim each, f32 accumulation, f16/bf16/f32 output, every extent a multiple of 8). `epilogue`
  says when a fusion's readers can run as the GEMM's epilogue (column regions along the last axis, aligned to
  16-byte vectors; not the two halves of the product). `GemmCodegen for Cuda` generates a CuTe `EPILOGUE`
  function, named `cuda_gemm_<hash>`, that receives vectors of the product staged in shared memory.
- `Config` (tile_m, tile_n, tile_k, warps, stages; `stages` 0 = as many as fit), `CONFIGS` (24 candidates),
  `untuned` (128x256/8 warps when its CTAs fill a wave, else 128x128/4 warps). Exposed to Python as
  `lumen.autotune.CudaGemmConfig`.
- `Gemm` (`Kernel`): tuned only at `TUNED_MACS` (1024³) multiply-adds or more. `run` transposes operands to
  A (.., m, k) / B (.., k, n) and calls the launcher installed in `runtime::kernel::cuda::GEMM`, with
  `Output::Product` for a plain matmul or `Output::Epilogue` carrying code from `codegen::devices::cuda::generated`.
- `ampere.py`: `AmpereGemm`, the SM80 kernel (mma.sync, cp.async), adapted from QuACK's GEMM classes.
- `launch.py`: compiles `AmpereGemm` per dtypes, majors, batching and tile against symbolic-extent tensors so
  Rust launches it with a plain `cuLaunchKernel`.

## triton

- `mod.rs`: `GemmCodegen for Triton` produces `GemmCode` (sizes `[batch, m, k, n]`, generated `prologue_a`,
  `prologue_b` jit loaders for the operands' strides, and an `epilogue` storing the tile with the fusion's
  elementwise ops, column regions included). The epilogue's body is `Product::stores`: from a value (a dot's
  product, or a split-K combine's sum), its regions, accumulator flags and expansion, it emits the per-region
  code with a caller's store, so `reduction::triton::split_k_combine` runs the same epilogue through its loop
  `STORE` helpers. The value itself, when it is an output besides its regions (a combine's sum read whole),
  is stored once before them. An output an update makes (`updated`: a `dynamic_update_slice` of a body input, a KV cache)
  is stored at the region values' slot (`slot`: their coordinates plus the clamped starts), the rest of the
  operand left as it is. `Config` (block_m/n/k, group_m, stages, warps), `CONFIGS` (7) with
  `UNTUNED` index; `row_gemm_configs` / `untuned_row_gemm` for whole-row tiles; `ROW_TILE` (256) caps a
  row held in one tile and `partial_tile` picks partial widths for wider rows. `gemv` is MPS's GEMV rule (its
  vector-lhs side: float accumulation, at most `GEMV_ROWS` (8) rows, the matrix contracted along its last dim);
  `gemv_configs` / `untuned_gemv` are the GEMV kernel's tiles (`block_m` the rows, a power of two). Python class
  `lumen.autotune.TritonGemmConfig`. `Gemm` (`Kernel`) runs dots, GEMM fusions, `dot_row` fusions and linear
  cross entropy.
- `launch.rs`: `gemm` generates prologues/epilogue once per tensor layout and calls the Python launcher set by
  `set_gemm`; `runs_gemv` picks the GEMV kernel for a matrix-vector product (`gemv`), unless a row epilogue or
  row halves need the GEMM; `hold_operands` starts an updated output from its operand unless it already lives in the
  operand's buffer (planned in place), for GEMM fusions and split-K combines alike; `is_gemm` decides which steps run here; `DOT_ROW` / `dot_partials` fusion names.
- `row.rs`: `row_epilogue` (a tile holds whole rows, so row reductions run on the accumulator and the product
  never reaches memory), `dot_row_fits`, and `partials_epilogue` (column partials for wider rows).
- `cross_entropy.rs`: runs `linear_cross_entropy` as the GEMM with a fixed per-tile max/sum epilogue, then a
  combine kernel.
- `gemm_kernel.py`: `gemm_kernel`, the fixed GEMM taking `PROLOGUE_A`/`PROLOGUE_B`/`EPILOGUE` as constexpr jit
  functions, and `dot_general`, its launcher.
- `gemv_kernel.py`: `gemv_kernel`, the matrix-vector product with the same generated prologues and epilogue (column
  halves included): each program holds every row and `block_n` columns, streams `block_k` of K a step and sums the
  products over K at the end. `dot_general` launches it with the GEMM's arguments.
- `linear_cross_entropy_kernel.py`: `cross_entropy_epilogue` and `cross_entropy_combine_kernel`.

## mps

- `mod.rs`: `plan_matmul`/`collapsed` collapse dot operands into one batched matmul and size its grid (fixed
  float, mid, small, integer and 64-bit tiles). `Algorithm` (Tiled, Mpp, Skinny, Gemv) and `GemmConfig`
  (bm, bn, bk, threads, cols, prefetch, algorithm), Python class `lumen.autotune.MpsGemmConfig`.
  Kernels: `Gemm` (GEMM fusions; `configs` filters candidates by epilogue threadgroup-memory fit, operand views,
  column groups, skinny/GEMV shape and MPP availability; tuned when the output has at least 64K elements or is
  skinny; `prepare` generates and compiles a variant per config, MPP ones by the MPP compiler; `accepts` keeps
  GEMV configs to matrix-vector bodies), `Matmul` (a lone `dot_general` on the precompiled
  `matmul_<in>_<accum>_<out>` kernel) and `LinearCrossEntropy`. `runs` says when a step uses `Gemm`.
- `codegen.rs`: `GemmCodegen for Mps` and helpers (`gemm_kernel`, `gemm_variant`, `gemm_regions_kernel`,
  `gemm_flags`/`EpilogueFlags`, `gemv_config`) emitting Metal that calls the matmul templates with a generated
  epilogue functor, including column regions and paired outputs.
- `row.rs`: matmuls fused with row reductions: single-tile `gemm_rows` and multi-tile row matmuls across
  `ROW_TILE` (64) column tiles, split into a partials pass and a combine (`tile_scratch`, `tile_passes`, `cut`).
- `tiled/` (`matmul_sg_kernel.metal`, simdgroup-matrix tiles; `config()`, `DEFAULT_CONFIGS`), `mpp/`
  (`kernel.metal`, Metal Performance Primitives; `mpp_configs`, `mpp_transposes`), `skinny/` (`kernel.metal`,
  few output rows splitting K; `skinny`, `skinny_configs`), `gemv/` (`kernel.metal`, matrix-vector;
  `gemv`, `gemv_configs`, `gemv_default`), `integer/` (`matmul_kernel.metal`, `matmul_wide_kernel.metal` for
  64-bit integers), `linear_cross_entropy/` (`kernel.metal`: `lce_partials_*` and `lce_combine`).
- The `.metal` files are compiled into the runtime library by `runtime/kernel/mps.rs` (`include_str!`).

## tpu

- `mod.rs`: `block` and `caps` choose block sizes; `kernel` builds the step's `codegen::devices::tpu::Kernel` for module
  `lumen.codegen.dot_general.tpu.gemm_kernel`.
- `codegen.rs`: `GemmCodegen for Tpu`, `Epilogue` (concatenated lhs, output slices, epilogue nodes) and `spec`,
  which fits blocks into an 88 MiB VMEM budget; only float32 and bfloat16 operands.
- `row.rs`: `kernel` for `dot_row` fusions; each block of up to 1024 rows computes the full `n`-wide output in
  VMEM and applies the row nodes.
- `gemm_kernel.py`, `row_kernel.py`: the Pallas `program(spec)` for each.

## Conventions

- Kernels never hard-code their epilogue: it is generated from the fusion body (CuTe/Triton jit functions, a
  Metal functor, or TPU spec `nodes`).
- Column regions come from `codegen/regions.rs` (`gemm_regions`, `gemm_expanded`); the GEMM dot of a body comes
  from `codegen/kinds.rs` (`gemm_dot`).
- CUDA and Triton GEMM launchers are Python functions registered when the Python extension module loads
  (`graph_compiler/python.rs`); running a GEMM with none installed fails with an error.
