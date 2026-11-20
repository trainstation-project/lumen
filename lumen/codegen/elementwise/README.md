# elementwise

Code generation for elementwise fusions: every output of a fused body computed per element (elementwise ops plus the layout ops that fold into index arithmetic). Each backend's codegen (`codegen/{cuda,triton,mps,tpu}.rs`) sends a fusion here when it matches no more specialized kind (GEMM, attention, reduction, row, transpose).

## Files

- `mod.rs` — the `ElementwiseCodegen` trait and the device submodules.
- `cuda/` — CuTe DSL backend.
- `triton/` — Triton backend.
- `mps/` — Metal backend.
- `tpu/` — Pallas backend.

## The trait

`ElementwiseCodegen::elementwise(body, by_value, views, schedule)` returns a backend's `Code` for the body. `by_value[k]` marks input `k` as passed as a scalar; `views[k]` is output `k`'s (element offset, strides) when it is written through a view. `Schedule` is what the code is generated for besides the body: MPS's `Tuning`, Triton's tensor layouts (`utils::triton::Tensors`), CUDA's `Layouts`, TPU's `()`. Each implementation is on the backend type from `codegen/utils` (`Cuda`, `Triton`, `Mps`, `Tpu`).

## Devices

- **cuda** — `mod.rs` generates a module whose `BODY` runs inside the fixed `ElementwiseKernel` in `kernel.py`: a grid-strided walk per segment (outputs of one element count), whole tiles without bounds checks then the tail. Inputs and outputs read at the walk's own index move as 16-byte vector fragments when every tensor is aligned. `Layouts` holds each tensor's strides and the elements it spans. The `Elementwise` kernel (Config `()`) runs elementwise fusions and output copies (reshapes); its code is generated per layout by `generated` (with `copy_body` for reshapes) and kept in the step's memo. The launch shape (threads, step bytes; `UNTUNED`, `shapes()`) is not a `Kernel` config: `graph_compiler/python.rs` times `shapes()` on the kernel's first launch when autotuning. `blocks` gives the grid size. `lumen.codegen.utils.cuda` compiles the module.
- **triton** — `mod.rs` generates a `BODY` jit function run by the fixed `elementwise_kernel` in `kernel.py`, one range of programs per grid segment. Loads and stores come from helpers generated for the tensors' layouts, so `views` is unused. This file also owns `LoopConfig` (block size, warps), `LOOP_CONFIGS` (block sizes 32..8192 times 4..32 warps), `loop_configs(lanes)` and `untuned_loop` (block 1024, 4 warps), which every Triton loop kernel shares. The `Loop` kernel runs every step that `codegen/utils/triton::loop_kernel` generated code for: elementwise, row, reduction, scan, updates fusions, cumsum, dynamic update slice and scatter-add. It generates helpers per tensor layout into the step's memo, copies the operand first for in-place ops, and returns no configs for a scatter-add. `fill` and `copy` are utility kernels that `Tensor` and steps call on CUDA tensors, cached per layout.
- **mps** — `mod.rs` generates a body functor that the fixed `elementwise` loop template (`kernel.metal`) runs per grid segment, storing outputs through their views. `Tuning::Elementwise { per_thread, group, contiguous }` sets how elements split across threads. `elementwise_tuning` picks a default that keeps the grid near 32K threads. `elementwise_tunings` gives the candidates: power-of-two elements per thread × threadgroup sizes 64..1024 × access order. The `Elementwise` kernel (Config `Tuning`) runs fusions that codegen kept tunable (`Tunable` holds what regenerating needs). Its `prepare` compiles every candidate variant together. It returns configs only for outputs in `TUNED_ELEMENTS` (64K..16M elements). `runs(step, tunable)` is how `codegen/utils/mps` decides whether a step gets this kernel; `tuning_of`/`tunings_of` give a fused body's default and candidate tunings. `PyMpsElementwiseConfig` exposes the configs to Python.
- **tpu** — `mod.rs` generates a whole `lumen.kernels.pallas` program (`program(device)`) for the fusion. A plain fusion computes nodes on blocks of the output's shape (`blocks` picks whole TPU tiles for every dtype under a byte budget). An indexed fusion (one with layout primitives) reads each value in blocks placed by `grid.rs`. `grid::plan` maps every value's dims onto the output grid and picks blocks that every operand reads as whole tiles. Nodes are rebuilt with `prims.bind`. An output can be written into a column window of a wider matrix. This folder has no `Kernel` struct: `elementwise(step)` wraps the result in the TPU `Kernel` spec (`codegen/utils/tpu`), which runs it with `Pallas`.

## Conventions

- The fixed loops (`kernel.py`, `kernel.metal`) never change per fusion; all per-fusion work arrives as the generated `BODY` or functor.
- Values are computed by the backend emitters in `codegen/emitter`, and loads and stores are addressed by `codegen/store`.
- A horizontal fusion's outputs of different element counts run as separate grid segments in one launch.
