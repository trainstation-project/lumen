# kernels

Kernel-language support that codegen builds on: a Pallas-style TPU kernel language (`pallas/`) and the CuTe DSL helpers the CUDA kernels use (`cuda/`). Nothing here decides what to fuse or which kernel to run; the device kernels themselves live in `lumen/codegen/<op>/<device>`.

## Files and folders

- `mod.rs` declares the Rust side, which is only `pallas`.
- `pallas/` is the Pallas TPU kernel language: a Rust kernel IR lowered to Mosaic MLIR plus a JAX-Pallas-like Python API. It has its own README.
- `cuda/` is a Python package of CuTe DSL (CUTLASS Python) helpers taken from QuACK and Tri Dao's code, with their copyright headers kept:
  - `copy_utils.py`, `layout_utils.py`, `cute_dsl_utils.py` (copies, layout transforms, `ParamsBase`), `pipeline.py` and `pipeline_checks.py` (async pipelines and their arrival-count checks), `tile_scheduler.py`, `reduce.py`, `fast_math.py`, `rounding.py` (rounding modes and stochastic rounding for GEMM epilogues), `utils.py`.
  - `dsl/` patches and extends the CuTe DSL on import: Pythonic tensor indexing, tensor helpers, mixed static/dynamic `if` in CuTe's AST preprocessor, and a system `ptxas` replacement.
  - `sync/` holds global-memory barriers for CuTe DSL kernels.

## How it connects

- The CUDA SM80 GEMM (`lumen/codegen/dot_general/cuda/ampere.py`) and flash attention forward and backward (`lumen/codegen/attention/cuda/ampere/`) import `lumen.kernels.cuda`.
- TPU kernels in `lumen/codegen/*/tpu` and `lumen/runtime/device_transfer/tpu` are written with `lumen.kernels.pallas`.

## Conventions

- Files in `cuda/` are vendored from QuACK; keep their copyright headers.
- `dsl/` works by patching CuTe when imported (`dsl/__init__.py` imports each patch module); the patches must be installed before a kernel that relies on them is compiled.
