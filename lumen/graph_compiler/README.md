# graph_compiler

The graph compiler turns a traced `Graph` into a runnable `Plan` for one device. It sits between tracing
and execution: graph -> the device's pass list -> codegen (`Device::lower`) -> plan.

## Flow

1. `compile` / `compile_with` take a graph, a `crate::Device` and `Options`.
2. When the graph moves values between host and device (`ToHost` / `ToDevice`), `stages::compile` splits
   it into alternating host and device stages, compiles each and joins them into one staged plan.
   Otherwise `stages::without_transfers` drops the transfer nodes and the graph is compiled as one plan.
3. `device_compile` picks the backend: CUDA runs on the CuTe DSL backend (`Cuda`), or on `Triton` when
   `use_triton_backend` is set; Apple GPUs on `Mps` (only when MPS is linked); TPUs on `Tpu`; everything
   else on `Cpu`.
4. `device::compile!` runs the device's pass list (`pipeline::Passes`, in `pipeline/<device>.rs`; every list
   starts with `preserve_precision`, `cse` and `simplify` without structural rewrites) on a `DeviceModule` for that
   device, then calls the device's `Device::lower`, which lives in `codegen` and builds
   the plan.

## Files

- `mod.rs` - `compile`, `compile_with`, `Options`, `plan_options`, plus helpers backends share:
  `elementwise` (which primitives elementwise fusion merges), `combined`, `constant`, `label`, `positions`.
- `device.rs` - the `Device` trait (`NAME`, `lower`), `DeviceModule<D>` (the graph plus state passes hand
  to later passes), the `compile!` macro and the CPU device.
- `pipeline/` - `Pipeline` (named, ordered list of passes) and each device's pass list (`Passes`).
- `stages.rs` - splitting host/device programs into stages; a host stage reads each value copied from the device
  through a `device_wait`, since the copy is started without waiting (`runtime/device_transfer/pending.rs`).
- `schedule.rs` - node schedulers used by `plan/compile.rs`: `list` and `dfs` (keep less memory live) and
  `overlap` (run collectives early and `device_wait`s, of collectives or of copies from the device, late).
- `remat.rs` - `rematerialize`: recomputes large values before later readers until the plan's workspace
  fits `memory_limit`; called from `plan/compile.rs`.
- `python.rs` - the Python `Plan` class, kernel launch capture for code dumps, and the hooks through which
  Triton and CUDA (CuTe) kernels, GEMMs and attention are launched from Rust.
- `__init__.py`, `_C.pyi` - Python shim and stubs for `lumen.graph_compiler.Plan`.
- `config/` - `CompilerConfig`, the flags every pass reads.
- `passes/` - every graph pass; see its README for the pipeline order.
- `tests/` - Rust unit tests of the shared passes and of the MPS and TPU passes.

## Invariants

- `DeviceModule` carries state between passes: `packed` and `interleaved` (weight blocks dot merging
  appends as the graph's last inputs), `rows` (normalization rows the diamonds pass finds for fusion) and
  `scalars` (input indices passed to kernels by value).
- `plan_options` treats packed blocks as parameters and keeps each member input alive while its block is
  read.
- A device runs only the passes its list adds, so the CPU's is just `unsupported` and `simplify`.
- With `dump_to` set, each pipeline run writes the graph before the first pass and after every pass to
  `<dir>/<run>.<pipeline>.<step>.<pass>.txt`. Each pass runs inside a profiler range `<pipeline>::<pass>`.
