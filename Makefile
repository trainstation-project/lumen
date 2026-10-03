# The jobs in .github/workflows/ci.yml call these targets; `make ci` runs
# every job locally.

export RUSTFLAGS := -D warnings

.PHONY: all ci test fmt fmt-check clang-format clang-format-check clippy miri clean

## ci: every CI job (rustfmt, clang-format, clippy, tests, miri), in that
## order.
ci: fmt-check clang-format-check clippy test miri

# The Rust tests are unit tests inside the crate (a `tests.rs` per module
# folder). They never enable `python`: with pyo3 compiled in, test binaries
# would link against libpython. The bindings are tested from Python
# (tests/test_*.py) instead.
#
# Device tests skip themselves when their device is missing, so one run
# covers every machine. MPS needs nothing extra: Metal is a default feature,
# compiled on macOS. CUDA is built where a GPU driver is visible (the crate
# and wheel with the `cuda` feature, the wheel with the CuTe DSL extras),
# so CPU-only machines skip the heavy extras.
HAS_CUDA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes)
CARGO_TEST_FEATURES := $(if $(HAS_CUDA),--features cuda)
comma := ,
MATURIN_FEATURES := $(if $(HAS_CUDA),--features python$(comma)cuda --extras cuda)
# The build Linux CI gets: the Metal backend is compiled only on macOS, so
# building without it here catches code that breaks when it is cfg'd out.
NO_METAL_FEATURES := --no-default-features --features cuda

## test: every test: the unit and doc tests, then the Python tests. CPU
## tests always run; MPS and CUDA ones run where the device is available.
test:
	cargo test --lib $(CARGO_TEST_FEATURES) -- --format=terse
	cargo test --doc
	maturin develop $(MATURIN_FEATURES) && python -m pytest tests -q -rs

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

# The C++/Objective-C++/Metal sources (the .mm shims and .metal kernels);
# style is the shared .clang-format (from lm-engine). Needs clang-format
# 21.1.6 on PATH (`pip install clang-format==21.1.6`).
CPP_SOURCES := $(shell find lumen -name '*.mm' -o -name '*.metal' -o -name '*.h')

clang-format:
	clang-format -i --style=file:.clang-format $(CPP_SOURCES)

clang-format-check:
	clang-format --dry-run --Werror --style=file:.clang-format $(CPP_SOURCES)

clippy:
	cargo clippy --all-targets --all-features
	cargo clippy --all-targets $(NO_METAL_FEATURES)

RUSTUP := $(or $(shell command -v rustup),$(wildcard $(HOME)/.cargo/bin/rustup))
# CI's platform, which Miri emulates on any host: on Apple Silicon, `half`
# converts f16 with inline assembly, which Miri cannot run.
MIRI_TARGET := x86_64-unknown-linux-gnu

miri:
	@test -n "$(RUSTUP)" || { echo "make miri: needs rustup with nightly miri: curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --default-toolchain none && ~/.cargo/bin/rustup toolchain install nightly --component miri,rust-src" >&2; exit 1; }
	MIRIFLAGS=-Zmiri-strict-provenance $(RUSTUP) run nightly cargo miri test --no-default-features --target $(MIRI_TARGET)

clean:
	cargo clean

update-precommit:
	uv run --extra dev --no-default-groups pre-commit autoupdate

style:
	uv run --extra dev --no-default-groups pre-commit run --all-files
