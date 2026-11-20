# autotune

Backend-independent kernel autotuning. After codegen lowers a graph to a device plan (`lumen/plan`), the autotuner times each tunable step's candidate configs on the device and sets the fastest one on the step. It also lets Python force a config in place of the autotuner's pick.

## Files

- `mod.rs` holds `Autotuner`, the `Timer` trait and the `KernelStep` trait. `Autotuner::tune_steps` groups the plan's steps by kernel, operand types and views, times each group's candidates on its first step, and gives every step in the group the winner.
- `forced.rs` keeps the stack of configs forced by `lumen.autotune.use` blocks (`push`, `pop`, `forcing`, `forced::<C>()`). Entering or leaving a block invalidates compiled plans so they are rebuilt with the new configs.
- `python.rs` registers `lumen.autotune._C`: the Python config classes (built with the `config_class!` macro around each kernel's own config struct) and `_push`/`_pop`.
- `__init__.py` exposes the config classes and the `use(*configs)` context manager.
- `_C.pyi` types the native classes: `CudaGemmConfig`, `CudaAttentionConfig`, `TritonGemmConfig`, `TritonAttentionConfig`, `TritonLoopConfig`, and, in a Metal build only, `MpsGemmConfig`, `MpsAttentionConfig` and `MpsElementwiseConfig` (otherwise `None` in Python). Each has `candidates()` returning what the autotuner times.

## How it connects

- Each backend's step type implements `KernelStep` (its kernel, name, candidates and forced config) and the backend provides a `Timer` that allocates scratch buffers and enqueues one run of a step. These live in `lumen/plan/step/{mps,cuda,triton}.rs`.
- Codegen builds the `Autotuner` (`lumen/codegen/utils`) once the plan's steps exist.
- The config structs themselves live next to their kernels in `lumen/codegen`; this folder only wraps them.
- Timing on or off comes from `lumen.config.compiler.autotune` (`lumen/config`). With it off, only forced configs apply.

## Conventions

- A step's timing is an average over a fixed number of runs after untimed warmup runs, and the device first runs for `WARMUP_SECONDS` to reach sustained clocks.
- Picks are cached process-wide, keyed by kernel family, device, primitive, operand types, views and output type, so later plans with the same step skip timing.
- Tuning runs outside the profiler.
- A forced config applies to every step of its kernel family, at any size, untimed, whether autotuning is on or not. An inner `use` block's config of a family replaces an outer one's. At most one config of each family type per block.
