# config

The global `lumen.config` object: process-wide runtime settings, with the compiler's settings under `lumen.config.compiler`. It holds no settings store of its own; it is a Python view over values kept elsewhere in Rust.

## Files

- `mod.rs` declares the Python bindings module.
- `python.rs` defines the `Config` singleton registered as `lumen.config._C.config`:
  - `static_allocator_bytes` reads and writes the accelerator region size in `lumen/allocator/config.rs`.
  - `compiler` returns the compiler config view (`CompilerConfig`) from `lumen/graph_compiler/config`.
  - `patch(**values)` returns a `Patch`; the shared `patch` helper is also used by `lumen.config.compiler.patch`.
- `__init__.py` re-exports `config` and defines `Patch`, the context manager and decorator that sets settings and restores the old values on exit.
- `_C.pyi` types the `Config` class.

## How it connects

- `lumen/__init__.py` exposes the singleton as `lumen.config`.
- Compiler flags (such as `autotune`, `flash_attention`, `merge_dots`, `safe_kernels`, `fused_linear_cross_entropy_chunk_size`) live in `lumen/graph_compiler/config`, which every device's compiler reads. Python code such as `lumen/autograd` reads them through `lumen.config._C.config.compiler`.

## Conventions

- `Patch` checks every name against the target's settings up front, so a typo raises `AttributeError` before anything changes. If setting one value fails, all values already set are restored.
- Patches nest: each `__enter__` saves its own copy of the old values.
- A new allocator size only affects allocators created after the change.
