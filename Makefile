# Local equivalents of the jobs in .github/workflows/ci.yml.

export RUSTFLAGS := -D warnings

.PHONY: all ci test fmt fmt-check clippy miri clean

ci: fmt-check clippy test

# Every feature but `python`: with pyo3 compiled in, Rust test binaries
# would link against libpython. The bindings are tested from Python.
RUST_TEST_FEATURES := --features cuda,mps

test:
	cargo test --all-targets $(RUST_TEST_FEATURES) -- --format=terse
	cargo test --doc $(RUST_TEST_FEATURES)
	maturin develop && python -m pytest tests -q

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features

## miri: needs rustup: `rustup +nightly component add miri`
miri:
	MIRIFLAGS=-Zmiri-strict-provenance cargo +nightly miri test

clean:
	cargo clean
