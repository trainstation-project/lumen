# Local equivalents of the jobs in .github/workflows/ci.yml.

export RUSTFLAGS := -D warnings

.PHONY: all ci test python-test fmt fmt-check clippy miri clean

ci: fmt-check clippy test

test:
	cargo test --all-targets --all-features -- --format=terse
	cargo test --doc
	cd python && maturin develop && python -m pytest tests -q

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
