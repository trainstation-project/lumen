# graph_compiler/config

The compiler's settings. Every device's compiler reads one `CompilerConfig`; each backend uses the flags it
needs and ignores the rest. Passes read it in `CompilerPass::configured` to say whether a device's pipeline runs
them (else the identity pass).

## Files

- `mod.rs` - the `compiler_config!` macro defining `CompilerConfig` (flags and defaults) and, with the
  `python` feature, its Python class `PyCompilerConfig`; the global store (`config`, `update`, `version`,
  `touch`) and `register`.
- `__init__.py` - re-exports `CompilerConfig` from the Rust extension.
- `_C.pyi` - Python stubs listing every flag with its default.

## How it is used

- Python reads and writes the global settings as `lumen.config.compiler.<flag>`. `reset()` restores the
  defaults; `patch(**values)` sets flags for a `with` block or a decorated function and restores them
  afterwards.
- `config()` returns a snapshot. `Options::default()` takes one, so a compile uses the settings current at
  that moment.
- Every `update` increments `version()` (`_version` in Python). Compiled-graph caches compare it to know
  when to recompile. `touch()` increments it without changing a flag, for example when an autotuning
  config is forced.
- `memory_limit()` returns `None` for 0 (no limit). `dump_to()` returns the dump directory.

## Flags

- Pattern passes: `merge_dots`, `flash_attention`, `fused_linear_cross_entropy` (+
  `fused_linear_cross_entropy_chunk_size`), `normalization_diamonds`, `split_k`, `linear_reordering`,
  `preserve_precision`, `neural_engine`.
- Fusion: `fuse` (master switch for fusion on MPS, CUDA and Triton), `reduction_epilogues`,
  `contraction_epilogues`, `multi_output_fusion`, `horizontal_fusion`, `online_softmax`, `row_cache`.
- Backend and runtime: `use_triton_backend`, `autotune`, `deterministic`, `cuda_graphs`, `fast_math`,
  `memory_limit`, `dump_to`.
- Flags whose results round differently from the traced program: `linear_reordering`, `fast_math` and
  `preserve_precision` (all off by default) and `online_softmax` (on by default).
- Also off by default: `neural_engine`, `deterministic`, `use_triton_backend`; `memory_limit` is 0 and
  `dump_to` and `fused_linear_cross_entropy_chunk_size` are unset.

Adding a flag means adding one line to the `compiler_config!` call and to `_C.pyi`.
