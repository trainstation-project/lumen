# Local equivalents of the jobs in .github/workflows/ci.yml.

export RUSTFLAGS := -D warnings

.PHONY: all ci test fmt fmt-check clippy miri clean

ci: fmt-check clippy test

# Every feature but `python`: with pyo3 compiled in, Rust test binaries
# would link against libpython. The bindings are tested from Python.
RUST_TEST_FEATURES := --features cuda,mps
# The build Linux CI gets: the Metal backend is compiled only on macOS, so
# building without it here catches code that breaks when it is cfg'd out.
NO_METAL_FEATURES := --no-default-features --features cuda

test:
	cargo test --all-targets $(RUST_TEST_FEATURES) -- --format=terse
	cargo test --all-targets $(NO_METAL_FEATURES) -- --format=terse
	cargo test --doc $(RUST_TEST_FEATURES)
	maturin develop && python -m pytest tests -q

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features
	cargo clippy --all-targets $(NO_METAL_FEATURES)

## miri: needs rustup: `rustup +nightly component add miri`
miri:
	MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test

clean:
	cargo clean
