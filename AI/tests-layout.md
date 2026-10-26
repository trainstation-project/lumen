---
name: tests-layout
description: Use whenever adding, moving or changing tests. Tests mirror the lumen/ package layout, are written once and parametrized by device instead of copied per backend, and share helpers instead of redefining them.
---

# Tests: one test per behaviour, laid out like lumen/

## Layout

1. **Mirror `lumen/`.** A test file lives in the `tests/` folder of the
   `lumen/` package it tests: `lumen/graph_compiler/` →
   `tests/graph_compiler/`, `lumen/kernels/` → `tests/kernels/`,
   `lumen/profiler/` → `tests/profiler/`. A top-level module
   (`lumen/nn.py`, `lumen/random.py`) is tested in `tests/test_<module>.py`.
   Name the file after the module or the topic it covers
   (`tests/graph_compiler/test_merge_dots.py` for `common/merge_dots.rs`).
2. **Unique file names.** `tests/` folders have no `__init__.py`, so every
   test file's name is unique across the tree (`test_dot_epilogues.py`, not
   a second `test_epilogues.py`).
3. **Put a new test where its code lives.** A test of a compiler pass goes
   with that pass's tests, a kernel's with its kernel. Don't add to a
   catch-all file; if no file fits, start one in the right folder.
4. **Shared helpers live in `tests/helpers.py`** (device params `MPS`,
   `CUDA`, `TPU`, `rand`, `meta`, the modules several files trace).
   Import them (`from helpers import MPS, CUDA, rand`); don't redefine a
   helper another file already has.

## No duplicated tests

5. **Parametrize by device, don't copy per backend.** A behaviour that
   doesn't depend on the backend (values against the CPU, how many steps a
   program fuses into, what merges) is one test with
   `@pytest.mark.parametrize("device", [MPS, CUDA, ...])` and the
   `triton_on_cuda` fixture, not a `test_mps_*` and a `test_triton_*`
   twin. Before writing a test, search `tests/` for one checking the same
   thing and add a parameter or a case to it instead.
6. **Backend-specific checks go inside the shared test.** Generated-source
   checks, block layouts and tuning details that only one backend has go
   under `if device == "cuda":` (or `"mps"`) in the shared test, after
   the checks every device shares.
7. **Expected differences are explicit.** When a device genuinely does
   something else (a fusion it doesn't do yet), the test states the
   per-device expectation (`2 if device == "mps" else 1`) and its
   docstring says why. Never weaken the check for every device.
8. **Single-device tests are the exception.** Keep a test to one device
   only when the behaviour exists only there (float64 has no MPS kernels;
   MPS stream internals), and name it after that (`test_mps_*`), not after
   a backend that merely ran it first.
9. **Device names in test names mean device-only.** A test that runs on
   several devices has no `mps_`/`triton_`/`cuda_` prefix.

## Checklist

- [ ] The test file sits in the `tests/` folder mirroring the `lumen/` code it tests.
- [ ] No other test already checks this; if one did, it was extended instead.
- [ ] Backend-neutral behaviour is one test parametrized over every device it runs on.
- [ ] Backend-only checks are under `if device == ...` in the shared test.
- [ ] Per-device differences are explicit and explained in the docstring.
- [ ] Helpers come from `tests/helpers.py`, not redefined.
- [ ] The suite was run on each device the tests cover (`make test` on a Mac; CUDA and TPU on their machines).
