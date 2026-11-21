# utils

Code shared by each backend's generators across ops: the backend marker types that the codegen traits are implemented on, dtype names, bit-exact constants, kernel naming, and the Python side that compiles and launches generated code.

## Files

- `mod.rs` — declares the backends. `mps` compiles only when MPS is linked (`lumen_mps_linked`).
- `cuda.rs` — `Cuda`, the CuTe DSL code generator type. `cute_type` gives a dtype's CuTe numeric type (`None` when there are no kernels for it). `unsigned` gives the unsigned type of a bit width. `constant` writes a value as its bits so it is exact in every dtype. It also holds CUDA's `codegen` (each planned step's kernel, generated per layout and kept in the step's memo) and its `Device` impl.
- `cuda.py` — Python side of the CUDA backend. `compile` compiles a generated module's `launch` for its tensors' CuTe types, 16-byte alignment and launch shape, loads it, and returns launch info (`CUfunction`, warps, shared bytes, parameter types) for Rust to launch. `compiled_cubin` dumps each compile into its own directory, because lumen kernels share names and concurrent processes would otherwise read each other's cubins.
- `triton/mod.rs` — `Triton`, the Triton code generator type. It provides `triton_type`, `params` (the parameter list of a generated kernel), the `fill` kernel source, `Tensors` (the layouts a template kernel's `BODY` is generated for), `body_function` (builds a template's `BODY(EXTRA, ...)` with its `LOAD{k}`/`STORE{k}` helpers), `bits`/`constant` (bit-exact scalars), `named` and `indented`. It also holds Triton's `codegen`: the generated loop kernel type (`Kernel`, `Template`, `Constexpr`), `loop_kernel`, and the `Device` impl.
- `triton/__init__.py` — Python side of the Triton backend. It compiles generated sources and caches them by name (`compiled`). `on_stream` runs launches on a given CUDA stream, which CUDA graph capture needs. `compiled_launch` returns a kernel's launch info for Rust. `template` and `launch` start template kernels and generated elementwise kernels. `scalar` passes runtime scalars, a float64 as its int64 bits.
- `mps/mod.rs` — `Mps`, the Metal code generator type. It provides `THREAD_ARGS` (thread position parameters), `formatted` (re-indents generated Metal), `named` (prepends load helpers and names a kernel by a hash of its source), `io_params` (buffer parameters), `metal_type`, `functor` (the prelude functor implementing a primitive) and `constant` (bit-exact via `as_type`). It also holds MPS's `codegen`: the kernel each step runs (`kernel`, `kernel_of`), the Metal kernels it generates (`Kernel`, `request`), `scratch_bytes`, and MPS's `Device` lowering.
- `mps/prelude.metal` — Metal source compiled ahead of every generated kernel (included by `runtime/kernel/mps.rs`). It contains dtype traits, XLA-semantics scalar ops (integer division and remainder, saturating float-to-int, NaN-propagating max/min, shifts, `reduce_precision`, `nextafter`, `pow`), the element functors, strided-offset helpers, type lists for instantiation, and simdgroup 8x8 fragment helpers. Philox lives separately in `codegen/random/mps/philox.metal`.
- `tpu/mod.rs` — `Tpu`, the TPU code generator type, and its `Device` impl, which plans buffers in the TPU tiled layout and generates each step's kernel. It also provides:
  - `fusible`: what a plain elementwise kernel runs.
  - `node`: the JSON `{name, params, args}` that Python kernels rebuild a primitive from.
  - `window`/`window_at`: an output written into a column window of a wider matrix.
  - `LANES` (128).
  The fusion and dot passes in `graph_compiler/passes` use these too.
- `tpu/kernel.rs` — TPU (Pallas) codegen, built only with the TPU linked and re-exported from `tpu`: the per-step `Kernel` spec, the kernel each step runs (`kernel`), and the `Pallas` and `Nothing` kernels.
- `tpu/lowering.rs` — lowers TPU kernels to Mosaic by calling `lumen.codegen.utils.tpu.pallas`, cached per (module, spec) and per device. It fails without Python.
- `tpu/pallas.py` — `lower` and `lower_source` trace a kernel module's `program(spec)` or a generated `program(device)` on a device's memory. `view` gives a ref into device memory in its TPU layout.
- `tpu/__init__.py` — `computed` and `per_row` replay an elementwise fusion's JSON nodes as primitives on kernel blocks. This is the only place where primitives are bound by name.

## Conventions

- Constants are always emitted from their raw bits, never as decimal literals, so every dtype is exact.
- Generated kernels are named after a hash of their source (`named`), so identical kernels share a name and a compile cache entry.
- The Rust side compiles and launches through these Python modules. If none is installed, CUDA and Triton launches fail with an error naming the module.
