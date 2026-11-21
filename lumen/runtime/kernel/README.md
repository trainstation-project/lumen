# runtime/kernel

Launching compiled kernels: each backend hands one kernel, its buffers and its arguments to its device queue
without waiting. Codegen generates and compiles the kernels; the `Kernel` implementations that plan steps hold
(`codegen/kernel.rs`) call these launchers from `Step::run`.

## Files

- `mod.rs` — the `Launch` trait: an associated `Args<'a>` (one launch in the device's form) and
  `launch(args)`, which queues it. It is implemented on each backend's codegen marker type (`Cuda`, `Triton`,
  `Mps`, `Tpu` from `codegen::devices`).
- `cuda/mod.rs` — `CudaLaunch` (compiled kernel, grid, runtime parameter values) launched with `cuLaunchKernel`
  straight from Rust on `stream::cuda::launch_stream()`, which is a CUDA graph's capture stream while one
  captures. Also the launchers Python installs once at import: `LAUNCH` (CuTe kernels), `GEMM` (with
  `Output`, the product or a fused epilogue's outputs) and `ATTENTION`.
- `cuda/driver.rs` — the CUDA driver entry points, loaded from `libcuda.so.1` with `dlopen`; `Value` (a runtime
  argument) and its Triton specialization `Class`; `Compiled` (a kernel Triton compiled for one class of launches)
  and its cache (`key`, `compiled`); `sm_count`; and `counted`, which tags the nodes a step adds to a capturing
  CUDA graph with the step's label.
- `triton.rs` — Triton launches as a CUDA kernel. Loop (elementwise) kernels go through the `Loop` launcher
  Python installs (`set_loop`), with their generated jit helpers (`Function`) and the compiled kernels they keep
  (`Launches`); `launch` computes the program count from block size, lanes, splits or segments. GEMM, attention
  and scatter-add launches belong to their codegen.
- `mps.rs` / `mps.mm` — Metal launches. `MpsLaunch` dispatches a kernel's cached pipeline on the MPS stream, or,
  inside `capture`, records it as `Captured`; `prepare` rewrites captured buffer pointers as offsets into the
  plan's allocations, and `replay` dispatches them against new ones. Also argument packing (`u32_arg`, `dims_arg`,
  ...), grids, `mark_event` for events, compiling generated Metal sources into the shared library (with Metal
  Performance Primitives templates when `mpp_available`), and `encode`, which launches a fused MPS step by its
  kind (`Launch`: attention, GEMM epilogue, row, reduction, transpose, elementwise, updates). `mps.mm` is the
  Objective-C++ side: library compilation, the pipeline cache and dispatch.
- `tpu.rs` — `TpuLaunch`: a kernel is lowered to Mosaic once (unless its step already did) and added to the open
  batch on its device; on flush the batch's kernels are chained into one StableHLO program over the device memory
  region, compiled once per launch sequence and run as one executable. Also SparseCore launches (`SparseCore`,
  `Indices`).

## Invariants

- A launch with a zero grid dimension is skipped.
- An MPS kernel takes at most `MAX_ARGS` (31) argument buffers; a replayed launch keeps its tensors alive only on
  the last launch of the step.
- A TPU launch carrying host data runs on its own after flushing the batch before it.
- Python-installed launchers are set once (`OnceLock`); later installs are ignored, and launching before one is
  installed errors.
- CUDA launches append two null scratch pointers after the parameters, as Triton's kernels expect.
