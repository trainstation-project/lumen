# kernels/pallas

A Pallas-style kernel language for TPUs, modelled on JAX's `jax.experimental.pallas` (`pl`/`pltpu`). Python code traces a kernel op by op into a Rust `Kernel` IR, which lowers to Mosaic MLIR text that the TPU runtime compiles. The TPU codegen writes its kernels with it, and users can call their own kernels from a traced program through `pallas_call`.

## Files

- `mod.rs` is the kernel IR: `Kernel` (typed variables, parameters, scratch and a body of `Block`s), `Type` (SSA `Value`, `Ref` in a memory `Space`, DMA `Semaphore`, or a `View` of device memory in the TPU tiled layout with its `Tiled` description), `Op`, and builder methods for loads and stores, strided access, slices and views, DMAs and indirect DMAs, and structured control flow (`begin_if`/`end_if`, `begin_while`/`end_while`, `begin_for`/`end_for`).
- `mosaic.rs` lowers a finished `Kernel` to Mosaic MLIR in its generic `"stable_mosaic.op"(...)` form (`lower`).
  Ops Mosaic lacks are built from others: unsigned max and min (and an unsigned max reduction) as signed ones of
  the values with their sign bits flipped, `expm1` as Kahan's `(exp(x) - 1) * x / log(exp(x))`.
- `python.rs` binds `Kernel` as `lumen.kernels.pallas._C.Kernel`.
- `__init__.py` is the Python kernel language: `Ref`, memory spaces (`VMEM`, `VMEM_SHARED`, `SMEM`), `SemaphoreType`, `ds`/`dslice`, control flow (`when`, `cond`, `switch`, `while_loop`, `fori_loop`, `scan`, `loop`), async copies, bit ops, `dot`/`dot_general`, and the entry points `kernel` (TensorCore) and `sparse_core_kernel` (SparseCore over a `(cores, subcores)` mesh).
- `pipeline.py` has `BlockSpec`, `Blocked`, `Element`, `BoundedSlice`, `program_id`, `num_programs`, `axis_index` and `emit_pipeline`, which double-buffers HBM to VMEM block copies over a grid.
- `call.py` has `pallas_call`: a kernel as one `pallas_call` node in a traced graph (extra outputs are `fusion_output` projections). The node stores a handle into `_CALLS`; the TPU plan later calls `program(spec)` to trace the kernel into Mosaic for the actual memory layout.
- `tests.rs` tests kernel building and the Mosaic text. `_C.pyi` types the native module.

## How it connects

- Ops inside a kernel reuse `lumen.graph.tracing.prims` and the tracer: `kernel` makes the `Kernel` the current "graph" with `tracer.tracing`, so `prims.bind` appends to it.
- `lumen/codegen/devices/tpu/lowering.rs` and `lumen/codegen/devices/tpu/pallas.py` drive the lowering once per spec and device; `view` there turns scalar-addressed rows of device memory into typed refs.
- Tiled views use `lumen/device/tpu/layout.rs`.

## Conventions

- A TensorCore kernel's parameters are fixed: an SMEM ref of int32 scalars, optional host rows, the device memory read, and the device memory written, each `uint32[rows, 1, 128]` (512-byte rows) in HBM. Tensors are addressed as row offsets into that one buffer, matching `lumen/allocator/tpu.rs`.
- The Python API follows JAX Pallas names and error messages where it can, so JAX kernels port with few changes.
- Calling a kernel-language op outside a kernel being traced raises `RuntimeError`.
- An `input_output_aliases` output starts as a DMA copy of its input.
