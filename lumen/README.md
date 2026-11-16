# lumen

The `lumen` package: a Rust crate (built as the `lumen._native` Python extension) plus Python packages around it. A user writes a function over tensors, `lumen.compile` traces it into a graph, and the graph is compiled into a static plan of hand-written or generated kernels for one device (CPU, Apple MPS, CUDA, Triton on CUDA, or TPU), which then runs on every call.

## Pipeline

tracing (`graph`, `functional`, `autograd`) -> graph (`graph`) -> graph compiler passes (`graph_compiler`) -> codegen (`codegen`, `kernels`, `autotune`) -> plan (`plan`) -> runtime (`runtime`, `allocator`, `device`)

## Top-level files

- `lib.rs` is the crate root: it declares the modules, the `library_name!`/`op_name!` macros that namespace op names as `lumen::<name>`, and re-exports `Tensor`, `Device`, `DType`, `Scalar`, `Storage`, `TensorOptions`, `Allocator` and `DataPtr`.
- `python.rs` builds `lumen._native` and registers each package's `_C` submodule (`lumen.<package>._C`) into `sys.modules`. A new package with Rust bindings adds its `register` function to the `PACKAGES` table here.
- `__init__.py` re-exports the user API at top level (`lumen.compile`, `lumen.grad`, `lumen.Tensor`, dtypes, factories, `lumen.config`, `lumen.Stream`, ...). `_native.pyi` and `py.typed` type the extension.
- `nn.py` holds `Module`, a frozen-dataclass tree of weights, and helpers that walk, map and hash such trees for `lumen.compile`.
- `random.py` holds the global counter-based random generator (a seed and a stream offset) and the traced random factories (`rand`, `randn`, `randint`) built on the `random_bits` primitive.

## Packages

- `graph/` - the traced `Graph`, its `Primitive`s, the Python tracer, `compile` and `prims`.
- `functional/` - torch-style functional ops (`lumen.functional`) traced into primitives, including attention and cross entropy.
- `named/` - named tensors (`lumen.named`): `NamedTensor` and ops taking axis names, broadcasting by name, resolved into positional `functional` ops while tracing.
- `autograd/` - JAX-style JVP and transpose rules, `grad`/`vjp`/`jvp`, custom `Function`s, run during tracing.
- `graph_compiler/` - frontend and per-device pass pipelines, compiler config (`lumen.config.compiler`), and lowering a graph to a plan.
- `codegen/` - per-op, per-device kernel generators (Metal, CUDA CuTe DSL, Triton, Pallas/Mosaic) and the shared `Kernel` trait.
- `kernels/` - kernel-language support: the Pallas TPU kernel language and the CuTe DSL helpers for CUDA kernels.
- `autotune/` - times candidate kernel configs per plan step and lets Python force configs.
- `plan/` - the compiled plan: ordered steps over inputs, outputs and one shared workspace, with a step type per device.
- `runtime/` - op dispatch, kernel launchers, streams, events, device transfers and waits.
- `allocator/` - device allocators: one static region per accelerator, carved by a first-fit sub-allocator.
- `device/` - the `Device` type, per-type backends, and the TPU PJRT shim and tiled layout.
- `tensor/` - the concrete strided `Tensor`, dtypes, scalars, parameters, NumPy and DLPack interop.
- `config/` - the global `lumen.config` object.
- `custom_op/` - user custom ops called from compiled code, Python kernel registration, user Metal kernels.
- `distributed/` - process groups, TCP store, device meshes and collectives (NCCL on CUDA, the store on CPU).
- `profiler/` - a `torch.profiler`-compatible profiler with device kernel timings and Chrome trace export.
- `reference/` - the element-by-element host interpreter used as ground truth and by CPU plans.
- `safetensors/` - safetensors save and load, mirroring `safetensors.torch`.
- `viz/` - HTML graph and plan pages and generated-code dumps.
- `stream/` - empty; streams live in `runtime/stream`.

## Conventions

- Programs always run inside `lumen.compile`; the Python ops trace primitives rather than compute eagerly.
- Python shims (`__init__.py`, `_C.pyi`) live in the same folder as the Rust they wrap, and each folder's bindings are registered as `lumen.<folder>._C` from `python.rs`.
- Device-specific code sits in `mps`, `cuda`, `triton`, `tpu` or `cpu` files or subfolders next to the shared code of the same feature.
