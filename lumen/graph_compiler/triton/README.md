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
| `mod.rs` (`fill`, `copy`, `reads_strided`) | `Function` (a prologue or epilogue: its parameter, name and source), passed with each launch: a loop gets a `LOAD{k}` for each input and a `STORE` from the tensors' layouts at run time, so a strided input is read in place. A CUDA tensor's `fill_` is a `STORE`, its `copy_` / `contiguous` from the same GPU a loop of one `LOAD` and one `STORE` (`Tensor::copy_`): on the GPU, never through the host. `reads_strided`: loops and GEMMs read any input as the view it is (`Plan::run_in` passes it, the executor hands it over with its strides). |
| `fusion.rs` | `LoopFusion` (`loop_fusion`): from the last node back, a fusible node (`codegen::triton::fusible`) joins the loop of its readers when they all are in one and it is no output of the program; a reshape no loop takes stays a view. Each group one `loop` fusion (`common/loops.rs`). |
| `gemm.rs` | The GEMM: `Gemm`, the hook `python.rs` installs (`triton_gemm`, calling `lumen.kernels.dot_general.triton.dot_general`); its 7 tilings (`codegen/triton/config.rs`); `run`, which launches a step with the tiling `TritonAutotuner` picked for it (`common/autotune.rs`, when `autotune` is on) or the default, and the prologues and epilogue `codegen/triton`'s `gemm` writes for its tensors' layouts. |
