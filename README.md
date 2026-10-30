# lumen

A minimal tensor library and graph compiler with a Rust core: a PyTorch-like
Python API, traced JAX/XLA-style into a graph of strict primitives, compiled
into a static memory plan, and run on lumen's own hand-written and generated
kernels (Metal on Apple silicon; a CPU reference everywhere; CUDA
allocation, streams and a CuTe DSL `fill_`).

```python
import numpy as np
import lumen
import lumen.functional as F

@lumen.compile
def layer(x, w):
    return F.relu(F.matmul(x, w, "float32", "float32"))

x, w = (lumen.from_numpy(np.ones(s, np.float32)).to("mps") for s in ((4, 8), (8, 16)))
y = layer(x, w)  # traced and planned on the first call per input signature, then run
print(lumen.to_numpy(y))
```

## Architecture in one flow

```
Python frontend        lumen.Tensor (data), lumen.functional, lumen.nn.Module
      |  lumen.compile(fn, device=...)
      v
Tracing                lumen/graph/tracer.py: TracedTensor records ops as
      |                strict primitives (lumen.graph.prims) into a Graph
      v                (the jaxpr analogue, lumen/graph/)
Autograd               lumen/autograd: JVP rules per primitive, linearize,
      |                transpose; grad / value_and_grad / vjp, and
      v                loss.backward() inside a compiled function
graph_compiler         lumen/graph_compiler: a pass pipeline per device
      |                (cse, simplify, then e.g. MPS dot merging, loop
      v                fusion, flash attention, horizontal fusion)
Plan                   graph_compiler/plan.rs: every value placed in one
      |                workspace allocated once; the steps to run
      v
Kernels                lumen/kernels/reference.rs (CPU reference: what each
                       primitive computes), lumen/kernels/*/mps + generated
                       Metal (lumen/codegen/mps), Core ML on the Neural
                       Engine (opt-in), CUDA (allocator, streams, fill_)
```

The Rust crate (`lumen/lib.rs`) is built with maturin into the native module
`lumen._native`, which registers each folder's bindings as `lumen.<folder>._C`;
the Python files in `lumen/` sit in the same folders as the Rust they wrap and
re-export them.

## Repository map

| Path | What it is |
| --- | --- |
| [`lumen/`](lumen/README.md) | The package: Rust core, Objective-C++/Metal shim, and Python API, one folder per subsystem (`graph`, `graph_compiler`, `autograd`, `ops`, `codegen`, `allocator`, `device`, `stream`, `tensor`, `profiler`, ...). |
| [`tests/`](tests/README.md) | Python tests of the bindings (pytest). Rust unit tests live inside the crate, a `tests.rs` per module. |
| [`examples/`](examples/README.md) | Runnable scripts: training steps, custom ops, horizontal fusion, a 2-layer transformer, the Neural Engine. |
| [`benchmarks/`](benchmarks/README.md) | Per-kernel GPU timing of the transformer example's training step. |
| [`.github/workflows/`](.github/workflows/README.md) | CI: rustfmt, clang-format, clippy, tests, Miri. |
| `build.rs` | Detects the CUDA toolkit (sets `cfg(lumen_cuda_linked)`) and compiles the Metal shim on macOS (`cfg(lumen_mps_linked)`). |
| `Cargo.toml` / `pyproject.toml` | Crate (features `mps` default, `cuda`, `python`) and maturin wheel (`lumen._native`, extras `cuda`, `dev`). |
| `Makefile` | Every build/check target; CI calls these. |
| `AGENTS.md` | Instructions for coding agents. |
| `.clang-format`, `.pre-commit-config.yaml` | C++/Metal style; Python hooks (autoflake, isort, black at 119 columns). |

## Build and test

Requires a Rust toolchain and [uv](https://docs.astral.sh/uv/), which makes
the checkout's `.venv` on Python 3.12 and installs everything into it:

```bash
curl https://sh.rustup.rs -sSf | sh   # Rust, if `rustc` is not on PATH
make test
```

| Command | What it does |
| --- | --- |
| `make install` | `uv sync` into `.venv` (made on Python 3.12 when missing): lumen built by maturin in release mode with the machine's device features, and its `test`, `dev` and device extras. Benchmarks (`dashboard/bench.py`, `serve.py`) must run on this build; a debug build is several times slower per call. |
| `make test` | `make install`, then `cargo test --lib`, `cargo test --doc` and `python -m pytest tests -q -rs -n auto` (a worker per TPU or CUDA chip, or per CPU core; then the `all_devices` tests serially), with the device features of the machine's hardware (CUDA, TPU) detected. **Always run this after a change** (AGENTS.md). |
| `make ci` | Every CI job locally: `fmt-check`, `clang-format-check`, `clippy`, `test`, `miri`. |
| `make fmt-check` | `cargo fmt --all -- --check`. |
| `make clang-format-check` | Checks the `.mm`/`.metal`/`.h` sources; needs `clang-format==21.1.6`. |
| `make clippy` | Clippy with all features, and again without Metal (`--no-default-features --features cuda`). |
| `make miri` | Crate tests under Miri (nightly, `x86_64-unknown-linux-gnu` target). |
| `make style` | Runs pre-commit, clang-format and `cargo fmt` in place. |
| `make clean` | `cargo clean`. |

The Makefile exports `RUSTFLAGS=-D warnings`. CUDA features are enabled
automatically when `nvidia-smi` is on `PATH`. MPS and CUDA tests skip
themselves where the device is missing, so one `make test` covers any machine.

## Conventions for contributors and agents

- **Run `make test`** when done (AGENTS.md).
- **Everything runs inside `lumen.compile`.** A `lumen.Tensor` holds data;
  only traced tensors have ops. Do not add eager, outside-a-trace paths.
- **Programs run exactly as traced.** No op changes a dtype the user did not
  ask for: operands must share a dtype, conversions are explicit
  (`.to(dtype)`); no hidden casts in compiler passes or kernels. Matmuls
  accumulate in float32 or wider.
- **Own kernels, own plan.** Graphs run through lumen's `Plan` and its
  hand-written/generated kernels, not a vendor graph runtime (e.g. not
  MPSGraph).
- **No safety checks in kernels.** Kernels read indices as given; clamping
  happens only behind `lumen.config.compiler.safe_kernels` (off by default).
- **Rust tests are in-crate** (`tests.rs` beside each `mod.rs`) and never
  enable the `python` feature; bindings are tested from `tests/`.
- **Tests are self-contained:** they copy example code rather than importing
  or executing files from `examples/`.
- **Unsafe code** documents itself: clippy's `missing_safety_doc` is answered
  with a `# Safety` section, never by allowing the lint.
- **Keep changes minimal:** reuse the existing APIs; no new helpers or
  wrappers a change does not need.
- Folder READMEs follow `lumen/device/README.md`: purpose, `## Files` table,
  key concepts, gotchas, related links.

## License

Apache 2.0 (`LICENSE`).
