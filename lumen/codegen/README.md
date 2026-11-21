# codegen

Codegen turns a compiled plan into runnable kernels. The graph compiler fuses the graph and lays out a `CpuPlan`
of steps; each backend's codegen then gives every step the kernel it runs (generating its source where needed),
and the autotuner (`lumen/autotune`) picks a config for each tuned kernel.

## Files

- `mod.rs`: declares the kernel families.
- `kernel.rs`: the `Kernel` trait every plan step runs, and `Runner`, the same kernel behind its config type.
- `kinds.rs`: tells fused bodies apart from the body alone: `gemm_dot`, `reduction_root`, `combine_root` (a
  split-K combine: the only reduction, a sum over at most `MAX_SPLIT` chunks of a body input, every output
  computed from it), `row_reductions`, `row_partials`.
- `regions.rs`: splits a GEMM's output into column regions by its readers so each region's epilogue fuses into the
  GEMM (`gemm_regions`, `gemm_pair`, `gemm_expanded`, `ROW_COLUMNS`); a split-K combine's value
  (`combine_value`, its sum through the rounding cast) splits the same way (`combine_regions`), readers of
  the whole value allowed besides its slices.

## Subfolders

One folder per kernel family, each with a device subfolder per backend that has the family:
`attention`, `dot_general` (GEMMs), `dynamic_slice`, `elementwise`, `factory`, `gather`, `layout`, `random`,
`reduction`, `row`, `scan`, `scatter_add`; and `fusion`, each backend's fusion (`Fuses`) for the graph compiler. Shared pieces live in `emitter` (per-backend statements computing a
body's values at an index) and `store` (strided load/store addressing). `devices` is each backend's home: its codegen
type, prelude and helpers, what it supports, and its `codegen` and `Device` impl, the dispatcher picking every step's
kernel across families (family-specific helpers live in their family).

Each backend implements the graph compiler's pass traits (`codegen/mod.rs`, plus `Attentions` and
`AttentionBackwards` in `passes/attention`) in the family file they belong to: attention traits in
`attention/<device>`; `SplitsK`, `DotLayouts`, `PartialRows`, `InnerProducts`, `MergesDots`, `FoldsDotReshapes`,
`FusesDots`, `LinearCrossEntropies` and `StridedOperands` in `dot_general/<device>`; `NormalizationRows`, `FusedRows`
and `FullRows` in `row/<device>`; `UpdateChains` in `dynamic_slice/<device>`; `Slots`, `Rechecks` and
`HorizontallyFuses` in `fusion/<device>`; `SwapsPairs` and `Gathers` in `emitter/<device>`. A TPU family module is
built without the TPU linked too, so the pipeline has its pass traits on every build; the parts that need the TPU
runtime are `#[cfg(lumen_tpu_linked)]`.

## The Kernel trait

A `Kernel` (in `kernel.rs`) has three associated types: the backend's `Step`, the `Buffers` it runs on (tensors,
an MPS encoding, or TPU raw pointers) and its `Config` (`()` for none). It provides:

- `configs(step)`: candidate configs to autotune; empty when the step isn't tuned.
- `accepts(step, config)`: whether a forced config can run the step.
- `prepare(step, configs)`: optional; compiles the variants for several configs together before any runs.
- `run(step, config, buffers)`: launches the step, untuned when `config` is `None`.

Every `Kernel` is also a `Runner<Step, Buffers>`, which hides the config type: candidates become `Tuned` values
named by the config's `Debug` text, the kernel's name (its Rust type name) keys tuning results, and a config
forced through `lumen.autotune.use` is honoured only when `accepts` allows it. Steps hold their kernel as an
`Arc<dyn Runner<..>>`; `autotune::KernelStep` reads candidates, forced configs and `prepare` through it.

## How each backend picks kernels

Each backend's `codegen(plan, options)` runs once per compiled plan and gives every step its kernel:

- CUDA (`devices/cuda`): a `DotGeneral` or GEMM fusion runs `Gemm`, the flash-attention forward fusion runs
  `FlashAttention`, everything else runs `Elementwise` (a reshape runs as a copy body). Elementwise and GEMM-fusion
  sources are generated per tensor layout and kept in the step's memo; any other op is an error.
- Triton (`devices/triton`): `loop_kernel` generates each step's loop kernel first; then attention runs
  `FlashAttention`, GEMMs `Gemm`, collectives `Collective`, waits `Wait`, fusions, cumsums, dynamic update slices
  and scatter-adds with a generated kernel `Loop`, and the rest `Unsupported`.
- MPS (`devices/mps`): generates and compiles Metal source for every pending fusion first, then picks `Gemm`,
  `FlashAttention` or the tunable `Elementwise` when they take the step, the generic fusion encoder for other
  fusions, else one kernel per primitive (`Reduction`, `Cumsum`, `ScatterAdd`, `LinearCrossEntropy`, `Matmul`,
  `Copy`, `DynamicSlice`, `Factory`, `RandomBits`, `HostWait`, `Unsupported`). `Gemm`, `FlashAttention` and
  `Elementwise` are shared by all steps of the plan.
- TPU (`devices/tpu/kernel.rs`): `kernel(step)` builds a Pallas spec by fusion name (`flash_attention*`, `reduction`, `dot_row`,
  `row`, updates, `elementwise`) or primitive; the step then runs `Pallas`, `Gather` (SparseCore when possible,
  else its TensorCore kernel) or `Nothing` (custom calls).

Each backend's `Device::lower` plans the fused graph, calls its `codegen`, then runs the autotuner's `tune_steps`
when autotuning is on or a config is forced (CUDA, Triton and MPS).

## Codegen traits

Each family's `mod.rs` defines a trait a backend's codegen type (`Cuda`, `Triton`, `Mps`, `Tpu` from `devices`)
implements: `AttentionCodegen`, `GemmCodegen`, `ElementwiseCodegen`, `ReductionCodegen`, `RowCodegen`,
`LayoutCodegen`, `DynamicSliceCodegen`, `ScanCodegen` and `RandomCodegen`. Each turns a fused body (or op) into
the backend's `Code`: Metal name and source, Triton source or helpers, CuTe DSL source, or a Pallas spec. The
`Kernel` structs that launch that code, with their configs and candidate lists, sit beside the trait impl in the
family's device subfolder.

## Conventions

- Inputs flagged in `by_value` are passed as scalars, not buffers.
- Output views are `(element offset, strides)` keyed by output index; kernels write through them.
- Hand-written kernels (`kernel.metal`, `kernel.py`) take fused work only through a spec or generated code, never
  through edits to the kernel.
