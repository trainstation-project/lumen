# `lumen/device`

Device abstraction: the small set of types and traits that answer *"which
piece of hardware is this tensor on, and can I actually use it?"*.

Everything that needs to talk about a compute device references this module;
nothing here allocates memory, launches kernels, or knows about tensors.
Each device type's allocators come from `lumen/allocator/`: `DeviceBackend`
requires `AllocatorBackend` (`allocator(index)`), which every device type
implements next to its allocator (`allocator/cpu.rs`, `meta.rs`, `mps/`,
`cuda.rs`), so `allocator_for(device)` is `device.backend().allocator(..)`.
It is a pure capability/label layer that the allocator, stream, op and
profiler modules build on top of.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `DeviceBackend` trait, the `Device` enum, its `Display` impl, and the `mod tests;` hook. |
| `cpu.rs` | `Cpu` / `CPU` — the always-present host backend. |
| `cuda.rs` | `Cuda` / `CUDA` — NVIDIA backend; queries `cudart` FFI when linked, otherwise reports zero devices. |
| `mps.rs` | `Mps` / `MPS` — Apple Metal backend; queries the `lumen_mps_shim` FFI when linked. |
| `meta.rs` | `Meta` / `META` — the placeholder/"no data" device, like `torch.device("meta")`. |
| `python.rs` | Python bindings (`python` feature): the `lumen.device` class and `resolve_device`, registered into `lumen._C`. |
| `__init__.py` | The `lumen.device` package: re-exports the class (`from lumen.device import device`); `lumen/__init__.py` rebinds `lumen.device` to it. |
| `_C.pyi` | Type stub for what `python.rs` defines, re-exported by `lumen/_C.pyi`. |
| `tests.rs` | Unit tests for backend naming, counts, `check()` and `is_available()` consistency. |

## The `DeviceBackend` trait

```rust
pub trait DeviceBackend: Sync {
    fn name(&self) -> &'static str;
    fn device_count(&self) -> usize;
    fn is_available(&self) -> bool { self.device_count() > 0 }
    fn check(&self, index: usize) -> Result<(), String> { .. }
    fn set_device(&self, _index: usize) {}
}
```

* `name()` — the lowercase type tag used in `Display` output and in error
  messages (`"cpu"`, `"mps"`, `"cuda"`, `"meta"`).
* `device_count()` — how many devices of this type exist. This is the one
  method each backend *must* implement; the two below have defaults derived
  from it.
* `is_available()` — `device_count() > 0`. Cheap "is this backend usable at
  all?" predicate.
* `check(index)` — validates a specific index, returning `Err(String)` with a
  human-readable message (`"cpu device 1 is not available (1 device(s) found)"`).
  The default implementation compares `index < device_count()`; backends
  override it only to add extra context (CUDA notes whether the crate was
  built without CUDA, MPS distinguishes "wrong index" from "no Metal at all").
* `set_device(index)` — makes `index` current on the *thread* for APIs that
  need it. No-op by default; only CUDA implements it.

Note the convention: `is_available()` answers "does this backend work here?"
while `check(index)` answers "is this *specific* device usable?" Tests rely
on the two agreeing for MPS.

## The `Device` enum

```rust
pub enum Device { Cpu(usize), Mps(usize), Cuda(usize), Meta(usize) }
```

`Device` is `Copy`, so it is passed by value everywhere. Each variant carries
an index even when the backend only ever has one device (`Cpu(0)`, `Meta(0)`),
which keeps the type uniform and lets generic code call `.index()` without
matching.

* `backend()` — maps a variant to its `&'static dyn DeviceBackend`.
* `index()` — extracts the index from any variant.
* `check()` — `self.backend().check(self.index())`.
* `Display` — formats as `"<name>:<index>"`, e.g. `"cuda:0"`. This
  intentionally lines up with `lumen.device` / `torch.device` strings on the
  Python side.

Because indices are part of the value, `Device::Cuda(0) != Device::Cuda(1)`
and hashing/equality are the derived ones (the enum derives
`Debug, Clone, Copy, PartialEq, Eq, Hash`).

## Backends

### CPU
Trivial: one device, index 0, always available. Any other index fails via the
default trait `check` implementation.

### Meta
Same shape as CPU: one device, index 0, always "available". Represents
shape-only tensors with no backing storage; availability here means the
*type* is usable, not that storage exists.

### CUDA
Availability is a **compile-time xor runtime** question:

* `lumen_cuda_linked` is set by `build.rs` (in `detect_cuda`, via
  `find_cudart`) only when the CUDA toolkit (`libcudart`) is actually found.
  The Cargo feature `cuda` being on is *not* enough — the feature may be
  enabled on a machine with no toolkit, in which case the backend compiles to
  a stub that reports `0` devices.
* When linked, `device_count()` calls `cudaGetDeviceCount` and maps a
  non-zero return code to `0` (errors are silently treated as "no devices",
  not panics).
* `check()` overrides the default only to append `"; lumen was built without
  CUDA"` in the unlinked case.
* `set_device()` (only compiled when linked) is the sole non-trivial
  implementation in the module. It reads the current device with
  `cudaGetDevice`, reads the current context with `cuCtxGetCurrent` (driver
  API, hence the second `extern` block linking `cuda`), and calls
  `cudaSetDevice` **only if** the device differs *or* no context is bound.
  The guard avoids clobbering an existing context — the same defensive
  dance PyTorch performs.

The FFI is fully gated behind `#[cfg(lumen_cuda_linked)]`, so `cuda.rs` still
compiles on machines with no CUDA headers or libraries.

### MPS
* Gated on `lumen_mps_linked`, set by `build_mps_shim` in `build.rs` when the
  Objective-C++ shim compiles (macOS only; the `mps` feature is a default
  feature but is a no-op elsewhere).
* `device_count()` calls `lumen_mps_available() -> i32` and returns 0 or 1 —
  MPS is a single logical device.
* `check()` is overridden so the error message distinguishes the two failure
  modes: index out of range while Metal exists (reports
  `"1 device(s) found"`), versus no Metal at all (`"lumen was built without
  Metal, or there is no Metal device"`). The default trait `check` could not
  express that difference.

## How the rest of the crate uses it

* `lumen/lib.rs` re-exports `Device` at the crate root (`use lumen::Device`).
* Allocation and dispatch: `lumen/allocator/mod.rs`, `allocator/cpu.rs`,
  `allocator/cuda.rs`, `allocator/mps/mod.rs` and
  `allocator/static_allocator.rs` take a `Device` and select a backend by
  matching on it. The CUDA allocator calls `CUDA.set_device(..)` before
  touching memory, which is why `set_device` exists.
* Ops and streams: `lumen/ops/mod.rs`, `lumen/ops/device_transfer/mod.rs`,
  `lumen/ops/wait/mod.rs`, `lumen/stream/cuda/mod.rs`, `lumen/stream/mps/mod.rs`.
* Profiling: `lumen/profiler/mod.rs`, `profiler/chrome.rs`, `profiler/summary.rs`,
  and `profiler/cupti.rs` (iterates `CUDA.device_count()`).
* Guarding GPU-only tests: `CUDA.is_available()` / `MPS.is_available()` are
  used to skip tests on machines lacking hardware (e.g. the allocator and
  tensor tests), and `Device::Cuda(CUDA.device_count())` is used as a
  deliberate "one past the end" index to test the error path.

## Python side

`lumen.device` in `python.rs` is the user-facing counterpart, modeled on
`torch.device`: parse a string (`"cuda:1"`), compare, hash, pickle, and repr.
Python code imports it from this package (`from lumen.device import device`);
the other folders' bindings import `resolve_device` from `python.rs`.
It is deliberately *not* the same type as the core `Device` — `resolve_device`
converts a Python device into a core `Device` and then calls
`allocator_for(device)`, which is where `check()` turns into a Python
`RuntimeError`. Because the core has no notion of a "current device" yet, an
unindexed Python device maps to index 0; it stays a distinct Python value
(`"cuda"` vs `"cuda:0"`), as in `torch`. Python behavior is covered by
`tests/device/` (`test_device.py`: the class; `test_tensor_devices.py`:
tensors on devices); Rust behavior by `lumen/device/tests.rs`.

## Adding a new backend

1. Add `foo.rs` with a `pub struct Foo;` and `pub static FOO: &dyn DeviceBackend = &Foo;`.
2. Implement `name` and `device_count`; override `check` only if the default
   message is misleading; override `set_device` only if the runtime needs it.
3. Gate any FFI behind a `cfg` flag emitted by `build.rs`, so the module
   still compiles where the hardware/toolkit is absent.
4. Add the variant to `Device` and arms to `backend()` / `index()` in `mod.rs`
   (the `match`es are exhaustive, so the compiler will list every site).
5. Extend `tests.rs` with a round-trip check like
   `every_device_has_a_backend`.
