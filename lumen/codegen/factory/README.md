# factory

Fixed kernels for the plan steps that create a tensor from nothing: a lone `full` (fill) or `iota`. Only MPS has
them. On other backends these ops are fused into elementwise kernels as index arithmetic.

## Files

- `mod.rs` declares the `mps` submodule, built only with `lumen_mps_linked`. This family has no codegen trait,
  because it generates nothing.
- `mps/mod.rs` defines the `Factory` kernel. It implements `codegen::kernel::Kernel` with no config, so it is
  never tuned. `codegen/utils/mps` picks it for `Full` and `Iota` steps.
  - A fill launches `fill_<bytes>` over the output, with the fill value as a constant argument.
  - An iota launches `iota_<dtype>` on a 3-D grid of `[inner, iota axis, outer]`.
  - Like other MPS kernels, it binds the step's inputs and then its output.
- `mps/kernel.metal`: the fixed kernels.
  - `fill_*` is instantiated per element byte width (`FOR_BYTES`), so it depends on byte size, not dtype.
  - `iota_*` is instantiated per numeric dtype (`FOR_NUMERIC`). It writes each element's grid `y` index,
    converted with `from_int<T>`.
  - `event_marker` is an empty kernel dispatched to mark a point in the command stream for profiling.
  - `build.rs` compiles this file into the primitive-kernel library, after `utils/mps/prelude.metal`, which
    provides `ELEMENTWISE_ARGS`, `FOR_EACH_ELEMENT` and the type lists.

## Conventions

- The iota axis must be the grid's `y` axis. The Rust launch builds the grid so that this holds, and the kernel
  reads its value from `gid.y`.
