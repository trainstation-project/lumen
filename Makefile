# The jobs in .github/workflows/ci.yml call these targets; `make ci` runs
# every job locally.

export RUSTFLAGS := -D warnings

.PHONY: all ci install test fmt fmt-check clang-format clang-format-check clippy miri clean

## ci: every CI job (rustfmt, clang-format, clippy, tests, miri), in that
## order.
ci: fmt-check clang-format-check clippy test miri

# The Rust tests are unit tests inside the crate (a `tests.rs` per module
# folder). They never enable `python`: with pyo3 compiled in, test binaries
# would link against libpython. The bindings are tested from Python
# (tests/, test_*.py) instead.
#
# Device tests skip themselves when their device is missing, so one run
# covers every machine. MPS needs nothing extra: Metal is a default feature,
# compiled on macOS. CUDA is built where a GPU driver is visible (the crate
# and wheel with the `cuda` feature, the wheel with the CuTe DSL extras),
# so CPU-only machines skip the heavy extras. A TPU is built where its chips
# are: a PCI device of Google's (vendor 0x1ae0) of the processing
# accelerator class (0x12; Google's virtual NIC and disk share the vendor).
# The crate and wheel get the `tpu` feature and the wheel the `tpu` extra
# (libtpu), which the TPU tests load (TPU_ENV; a TPU_LIBRARY_PATH already
# set wins).
HAS_CUDA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo yes)
# A Trainium machine has Neuron devices (/dev/neuron*); the crate and wheel
# get the `trainium` feature, linking libnrt from the Neuron SDK.
HAS_TRAINIUM := $(shell ls /dev/neuron0 >/dev/null 2>&1 && echo yes)
HAS_TPU := $(shell for d in /sys/bus/pci/devices/*; do [ "$$(cat $$d/vendor 2>/dev/null)" = 0x1ae0 ] && case "$$(cat $$d/class)" in (0x12*) echo yes; break;; esac; done)
comma := ,
empty :=
space := $(empty) $(empty)
DEVICE_FEATURES := $(strip $(if $(HAS_CUDA),cuda) $(if $(HAS_TPU),tpu) $(if $(HAS_TRAINIUM),trainium))
CARGO_TEST_FEATURES := $(if $(DEVICE_FEATURES),--features $(subst $(space),$(comma),$(DEVICE_FEATURES)))
MATURIN_FEATURES := $(if $(DEVICE_FEATURES),--features $(subst $(space),$(comma),python $(DEVICE_FEATURES)) --extras $(subst $(space),$(comma),$(DEVICE_FEATURES)))
TPU_ENV := $(if $(HAS_TPU),TPU_LIBRARY_PATH="$${TPU_LIBRARY_PATH:-$$(python -c 'import libtpu, os; print(os.path.join(os.path.dirname(libtpu.__file__), "libtpu.so"))')}")
# The build Linux CI gets: the Metal backend is compiled only on macOS, so
# building without it here catches code that breaks when it is cfg'd out.
NO_METAL_FEATURES := --no-default-features --features cuda

## install: lumen built in release mode and installed into the active
## virtualenv (maturin develop --release), with the device features of the
## machine's hardware. Benchmarks and the dashboard measure this build.
install:
	maturin develop --release $(MATURIN_FEATURES)

## test: every test: the unit and doc tests, then the Python tests. CPU
## tests always run; MPS, CUDA and TPU ones run where the device is
## available. The release wheel is installed first (`install`): it installs
## the device extras the tests load (libtpu), and never leaves a debug build
## behind for a benchmark to measure. On a TPU machine where lumen sees no TPU, it fails
## rather than skip the TPU tests. The Python tests run in parallel
## (pytest-xdist): a worker per TPU or CUDA chip, each seeing its own, or
## a worker per CPU core (tests/conftest.py); then, serially, the tests
## needing every device (`all_devices`).
test: install
	$(if $(HAS_TPU),$(TPU_ENV) python -c "import lumen$(comma) sys; sys.exit(0 if lumen.tpu.is_available() else 'TPU hardware found but lumen sees no TPU: is libtpu installed (pip install libtpu) or TPU_LIBRARY_PATH set?')")
	$(TPU_ENV) cargo test --lib $(CARGO_TEST_FEATURES) -- --format=terse
	cargo test --doc
	$(TPU_ENV) python -m pytest tests -q -rs -n auto -m "not all_devices"
	$(TPU_ENV) python -m pytest tests -q -rs -m all_devices

fmt-check:
	cargo fmt --all -- --check

# The C++/Objective-C++/Metal sources (the .mm shims and .metal kernels);
# style is the shared .clang-format (from lm-engine). Needs clang-format
# 21.1.6 on PATH (`pip install clang-format==21.1.6`).
CPP_SOURCES := $(shell find lumen -name '*.mm' -o -name '*.metal' -o -name '*.h')

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
	clang-format -i --style=file:.clang-format $(CPP_SOURCES)
	cargo fmt --all
