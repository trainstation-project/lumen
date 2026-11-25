# passes/unsupported

The `unsupported` pass, first in every device's list: it fails compilation on the first op the device has no kernel
for (or the CPU no interpretation for), naming the op.

## Files

- `mod.rs` - the `Supports` trait (`check(graph)`) and the `Unsupported` pass running it.
- `cpu.rs`, `cuda.rs`, `mps.rs`, `triton.rs`, `tpu.rs` - each device's `Supports`: the CPU (no Pallas kernels), CUDA
  (ops and dtypes the CuTe DSL backend has kernels for), MPS (float64 and float8, bfloat16 transcendentals, dots of
  two dtypes), Triton (float16 and bfloat16 transcendentals) and TPU (float16, cumsum, scatter-add, random bits).
