# `lumen/graph_compiler/triton`

The CUDA compiler behind `use_triton_backend` (off by default): a program
on a CUDA device compiles here, and its plan's executor is the CUDA device,
which launches each step through Triton from Python. Without torch: Triton
launches on the device's default stream and the autotuner times with
lumen's CUDA events.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `compile`: the pipeline (`simplify`, then `loop_fusion`), the CPU's planning (`cpu::plan`), each loop fusion's kernel generated (`codegen/triton`) into `Plan::triton`, and the GEMMs tuned with `autotune`. `run`, what the executor calls for a step: a `dot_general` as the GEMM, a loop fusion as its kernel; any other step errors, as it has no Triton kernel yet. `Kernel` (name, source, which inputs are runtime scalars) and `Loop`, the hook `python.rs` installs (`triton_loop`, calling `lumen.codegen.triton.launch`). |
| `fusion.rs` | `LoopFusion` (`loop_fusion`): from the last node back, a fusible node (`codegen::triton::fusible`) joins the loop of its readers when they all are in one and it is no output of the program; a reshape no loop takes stays a view. Each group one `loop` fusion (`common/loops.rs`). |
| `gemm.rs` | The GEMM: `Gemm`, the hook `python.rs` installs (`triton_gemm`, calling `lumen.kernels.dot_general.triton.dot_general`); `CONFIGS`, its 7 tilings; `run`, which launches a step with the tiling `TritonAutotuner` picked for it (`common/autotune.rs`, when `autotune` is on) or the default. |
