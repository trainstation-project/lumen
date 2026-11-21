# attention

Flash attention for every backend. The attention passes (`graph_compiler/passes/attention`) match an attention in
the graph and fuse it, with its score ops, dropout, epilogue and log-sum-exp outputs, into a `flash_attention`
(forward) or `flash_attention_backward` fusion. This folder holds the hand-written kernels those fusions run and the
code generated per fusion so the fixed kernels can read and write its operands in their traced layout.

## Files

- `mod.rs` — the `AttentionCodegen` trait each backend implements: `forward` and `backward` code for a matched
  `Attention` / `Backward` laid out as `Strides`, `score` (the traced score ops applied to the float32 `Q K^T`
  products) and `dropout`. `Code` differs per backend: a Metal kernel's name and source, Triton jit helpers, or a
  TPU kernel kind and JSON spec. It also defines `TiledAttentions` (`attention_layouts`, TPU's).
- `cuda/` — the CuTe DSL (SM80) forward.
  - `mod.rs`: `supports` decides which matched attentions the kernel runs (16-bit floats written in their own dtype,
    head dims of 16-byte rows up to 256, no dropout or runtime causal offset, float32 LSE, operands in 16-byte rows).
    `prepare` turns a fusion body into a `Forward` (its `Operand`s as (batch, rows, heads, head dim) views, grid,
    heads per KV head, causal shift, score ops). `Config` is one `AmpereAttentionForward` candidate (CTA tile,
    stages, warps, Q in registers); `CONFIGS` are the candidates, `fits` checks one against the shared-memory budget
    and head dims, `untuned` picks the default. The `FlashAttention` Kernel tunes only attentions of at least
    `TUNED_MACS` multiply-adds.
  - `launch.py`: compiles a forward fusion through `ampere.forward.AmpereAttentionForward` once per operand layout,
    score ops, causal diagonal and config, against fake tensors of those exact shapes and strides, so Rust launches
    it as a plain `cuLaunchKernel` through `runtime::kernel::cuda::ATTENTION` (installed from Python with
    `set_attention`).
  - `ampere/forward.py`, `ampere/backward.py`: FlashAttention-2 kernels in the CuTe DSL, ported from
    FlashAttention's SM80 code. Codegen only routes forward fusions to CUDA; nothing in Rust launches the backward
    kernels.
- `triton/` — Triton flash attention, forward and backward.
  - `kernel.py`: the fixed kernels `forward_kernel` (writes each row's LSE), `dkdv_kernel` and `dq_kernel`, plus
    the `attention` launcher. They reach operands only through generated helpers (`LOAD_*`, `STORE_*`, `SCORE`).
  - `mod.rs`: `AttentionCodegen for Triton` generates those jit helpers per fusion (tile loads and stores through the
    matched layout, the output store running the body's elementwise epilogue, atomic dQ adds, score ops); dropout
    is unsupported. `AttentionConfig` and `ATTENTION_CONFIGS` are the tuning candidates, `untuned_attention` the
    default (64x64, or 32x32 when a head row exceeds 256 bytes). `FlashAttention` is the Kernel for both forward
    and backward steps.
  - `launch.rs`: `is_attention` (used by `codegen/devices/triton` to pick the kernel), per-layout `Prepared` state, and
    `attention`, which calls the Python launcher installed by `set_attention`. Operands whose last dim isn't
    contiguous are copied contiguous first.
- `mps/` — Metal flash attention.
  - `forward_kernel.metal` (tiled simdgroup-matrix prefill kernel and per-query decode kernel),
    `forward_mpp_kernel.metal` (forward on Metal Performance Primitives `matmul2d`, compiled only into the MPP
    library), `backward_kernel.metal` (dK/dV per key block with optional atomic dQ, and a dQ kernel). These are
    templates in the Metal prelude.
  - `mod.rs`: `AttentionCodegen for Mps` generates a kernel per fusion for its operands' layout, including dropout
    (Philox) and the epilogue. `Tuning` selects 8x8 matrices or MPP, simdgroups per threadgroup and key block;
    `tuning` is the untuned choice, `tunings` the candidates (none for decode or below `TUNED_PAIRS` query-key pairs
    a head), `fits` / `mpp_fits` the constraints. `backward_block` and `backward_block_with_dq` size the backward's
    query blocks to threadgroup memory. `FlashAttention` is the forward Kernel: `prepare` generates and compiles a
    variant per tuning. `runs` tells `codegen/devices/mps` which steps it takes, and `attention_reads_strided` which operands
    it strides through; backward fusions run through the generic
    fusion encoder.
- `tpu/` — Pallas flash attention, modeled on splash attention.
  - `forward_kernel.py`, `backward_kernel.py` (dQ, and dK with dV): Pallas programs taking a JSON spec;
    `utils.py` holds their shared index arithmetic, head/block specs, causal masking and online-softmax layouts.
  - `config.rs`: `blocks` picks query and key block sizes (multiples of 128 lanes, capped at 1024 / 4096).
  - `mod.rs`: `AttentionCodegen for Tpu` builds the spec: each operand checked to be read or written as whole tiles
    per head, epilogues limited to reshapes, sums and casts reading scalar inputs; only float32 and bfloat16, no
    dropout. `step_kernel` is called from `codegen/devices/tpu` for fusions named `flash_attention*` and runs under the
    generic `Pallas` Kernel.

## Conventions

- The kernels are fusion-agnostic: everything fused (score ops, epilogue, layouts, dropout) arrives as generated
  helpers, a generated Metal kernel, or a spec. A new epilogue never edits a kernel file.
- Per-layout work (CUDA compile, Triton helpers, Metal variants) is prepared once per operand layout and cached on
  the step.
- Backward kernels that add dQ atomically need a zeroed float32 accumulator.
- Causal masks are bottom-right aligned and may be shifted by an offset; query/key counts need not be tile
  multiples.
