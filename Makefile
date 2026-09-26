# The jobs in .github/workflows/ci.yml call these targets. `make ci` runs
# every job locally, with the full `test` where GitHub runs only `test-cpu`.

export RUSTFLAGS := -D warnings

.PHONY: all ci test test-cpu test-cuda test-mps fmt fmt-check clippy miri clean

## ci: every CI job (rustfmt, clippy, tests, miri), in that order; runs
## all tests (CPU, CUDA, MPS), where GitHub CI runs only the CPU ones.
ci: fmt-check clippy test miri

# Each test runs in exactly one of the targets below. Rust tests never
# enable `python`: with pyo3 compiled in, test binaries would link against
# libpython. The bindings are tested from Python instead.

# Device-independent integration tests: every tests/*.rs except the
# per-device files.
CPU_TESTS := $(filter-out cuda mps,$(basename $(notdir $(wildcard tests/*.rs))))
# The build Linux CI gets: the Metal backend is compiled only on macOS, so
# building without it here catches code that breaks when it is cfg'd out.
NO_METAL_FEATURES := --no-default-features --features cuda

test: test-cpu test-cuda test-mps

## test-cpu: library unit tests, device-independent integration tests and
## doc tests (built without Metal), then the Python tests.
test-cpu:
	cargo test --lib $(addprefix --test ,$(CPU_TESTS)) --no-default-features -- --format=terse
	cargo test --doc --no-default-features
	maturin develop && python -m pytest tests -q

## test-cuda: tests/cuda.rs. Runs the caching logic against mock backends;
## the real CUDA backend is compiled too when build.rs finds cudart.
test-cuda:
	cargo test --test cuda $(NO_METAL_FEATURES) -- --format=terse

## test-mps: tests/mps.rs against the real Metal device (macOS; elsewhere
## the file compiles to nothing).
test-mps:
	cargo test --test mps --features mps -- --format=terse

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features
	cargo clippy --all-targets $(NO_METAL_FEATURES)

## miri: the tests under Miri, which catches UB (misaligned access,
## use-after-free, leaks) in the allocator's raw-pointer code. Built without
## Metal, as on Linux CI: Miri cannot call into the Metal shim. Needs rustup:
## `rustup +nightly component add miri`.
miri:
	@command -v rustup >/dev/null || { echo "make miri: needs rustup (https://rustup.rs), then: rustup +nightly component add miri" >&2; exit 1; }
	MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test --no-default-features

clean:
	cargo clean
