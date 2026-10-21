# `lumen/allocator`

Memory allocation for every device. This module owns the `Allocator` trait,
the RAII handle that returns memory when it dies (`DataPtr`), and the
per-device implementations — including the *static allocator*, one big
up-front region per device that compiled execution allocates out of without
touching the driver.

It is the layer between `lumen/device` (which says *what* hardware exists) and
`lumen/tensor/storage.rs` (which asks for bytes on a device and holds the
result).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | The `Allocator` trait, `DataPtr`, `allocator_for`, and `Region` — the first-fit bookkeeping over one reserved range of a device's memory, which the MPS and CUDA allocators own. The heart of the folder. |
| `config.rs` | Process-wide tunables (currently just the static allocator size), set from Python via `lumen.config`. |
| `cpu.rs` | `CpuAllocator` — 64-byte-aligned `alloc`/`dealloc`, plus CPU memory profiling. |
| `meta.rs` | `MetaAllocator` — hands out a dangling pointer; no storage exists. |
| `cuda.rs` | `CudaAllocator`: a `Region` over one `cudaMalloc` per device (FFI), and the per-device registry. |
| `mps/mod.rs` | `MpsAllocator`: a `Region` over one Metal buffer (the shim's FFI), and its shared instance. |
| `mps/mps_shim.mm` | Objective-C++: `MTLBuffer` creation, the pointer→buffer map, and device limits. |
| `tpu.rs` | `TpuAllocator`: a `Region` over one byte-linear PJRT buffer per TPU (`lumen/device/tpu`), in whole rows (`ROW`, 512 bytes); its per-device registry, `copy_to_host` (reads at an offset), `run` (a TPU program on the buffer) and `rows`. Only with `cfg(lumen_tpu_linked)`. |
| `python.rs` | `lumen.config` (`Config` pyclass), plus the `compiler` sub-config. |
| `__init__.py`, `_C.pyi` | Python-side re-export and type stubs. |
| `tests.rs` | CPU, static-allocator, device-crossing, CUDA, MPS and TPU test modules. |

## The `Allocator` trait

```rust
pub trait Allocator: Send + Sync {
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize);
}
```

The one abstraction: *who gives this memory back*. Three allocators implement
it, one per device — `CpuAllocator`, `MetaAllocator`, `MpsAllocator`, `CudaAllocator` — plus DLPack's
adopted buffers (`lumen/tensor/dlpack.rs`), which free through the
producer's deleter.

* `allocate` returns raw, uninitialized memory of at least `nbytes`, or
  panics (out-of-memory is a panic here, not a `Result`).
* `deallocate` is `unsafe` because it trusts the caller to pass back a live
  block from the same allocator with the same size. It is only ever called
  from `DataPtr`'s `Drop` and from the tests.
`allocator_for(device)` is the public entry point. It calls `device.check()`
first, then picks the device's one shared allocator with a `match`: the CPU's,
the meta device's, the Metal device's (`mps::allocator()`) or CUDA device
`i`'s (`cuda::allocator(i)`), each created on first use. Checking
first means so an unavailable device produces the device module's error string
(`"CUDA device 0 is not available (0 device(s) found)"`) rather than a panic
from inside an allocator. `Storage::new` turns that `Err` into a panic;
`device/python.rs` turns it into a Python `RuntimeError`.

## `DataPtr`

```rust
pub struct DataPtr { ptr: NonNull<u8>, nbytes: usize, allocator: Arc<dyn Allocator> }
```

`DataPtr` is the owning handle: it holds the `Arc` to the allocator that
produced the block, so the allocator (and for the static allocator, the whole
region) cannot be dropped while any allocation from it is alive. Its `Drop`
calls `deallocate(ptr, nbytes)`, which is the only place that happens.

It is `unsafe impl Send + Sync` — sound because the raw pointer is never
dereferenced in this module, only offset and handed out; synchronization is
the responsibility of whatever code does dereference it.

* `allocate(allocator, nbytes)` reserves through the allocator.
* `from_raw_parts(ptr, nbytes, allocator)` (`unsafe`) wraps memory this
  module did not allocate, used by the DLPack import path
  (`lumen/tensor/dlpack.rs`) where freeing goes back to the producer's
  deleter.
* `as_ptr()` / `nbytes()` / `allocator()` are the accessors;
  `nbytes()` is the *requested* size, which the static allocator rounds up
  internally but reports back unchanged.

`Storage` wraps a `DataPtr` and exposes `device()`, `allocator()` and
`nbytes()`; `Storage::with_allocator` lets callers (tests, DLPack) supply
their own allocator instead of `allocator_for`.

## The static allocators and `Region`

The MPS and CUDA allocators are why this module is not just a thin `cudaMalloc`
wrapper. Their shared bookkeeping is `Region` — not an allocator itself, just
the free list over a range its owner reserves.

**Why it exists.** Compiled execution lays buffers out up front and reuses
them run after run. Calling `cudaMalloc` per tensor would be far too slow and
would fragment device memory. Instead each device reserves **one region**
(one `cudaMalloc`, one `MTLBuffer`) the first time anything allocates, and
every subsequent allocation is pure host-side bookkeeping — no driver call at
all.

**Data structures.** A `Mutex<State>` holds:

* `region: Option<Region>` — reserved lazily on the first allocation, never
  before. `Region` wraps a `NonNull<u8>` and is `unsafe impl Send` (the
  pointer is only offset, never dereferenced here).
* `free: BTreeMap<usize, usize>` — offset → length of free blocks. Offsets
  and lengths are multiples of the alignment and no two blocks touch.
* `live: BTreeMap<usize, usize>` — offset → requested bytes of outstanding
  allocations, plus `live_bytes`.

**Algorithm.** Allocation rounds the request up to the alignment
(`nbytes.next_multiple_of(alignment)`) so every block starts aligned, then
takes the **lowest free block that fits** (first fit), splitting off the
remainder. Freeing returns the block and **merges it with free neighbours**
on both sides, which is what makes a training loop that keeps its inputs
alive reuse the same bytes every iteration instead of growing. A zero-byte
allocation takes no block at all and returns any aligned offset.

**Alignment.** `ALIGNMENT` = 256, matching `cudaMalloc`'s and Metal's
guarantees. `Allocator::allocate_aligned` asks for more (the TPU's: a tiled
tensor's tile row, a plan's workspace its values'): `Region::take_aligned`
takes the first free block holding an aligned start, leaving the gap before
it free.

**Laziness and teardown.** `Region::new(device, capacity)` reserves nothing;
the first `take` reserves the range through its owner's `reserve` closure
(`MpsAllocator`: the Metal shim; `CudaAllocator`: `cudaMalloc`; the CPU's
allocator in tests). Dropping the owner gives the range back: it takes the
start with `region.base()` and frees it with its FFI. Each device's allocator
is one shared `Arc` and every `DataPtr` keeps its allocator alive, so the
range outlives all of its allocations.

**Accounting.** `capacity()`, `used()` (bytes to the end of the highest live
allocation) and `live_bytes()` are exposed for tests and diagnostics.

**Out of memory** is a panic that names every useful number: device,
requested bytes, free bytes, capacity, largest free block, and the exact knob
to raise (`lumen.config.static_allocator_bytes`). Before panicking it waits
for the device's work in flight (`synchronize(device)`: MPS's stream) once and
retries — freed memory can still be held by queued Metal work.

The allocator also reports every allocation and free to the profiler
(`crate::profiler::report_memory`) with live/capacity totals. The call is
unconditional but cheap: `report_memory` returns immediately unless the
current session has the memory flag set.

## Per-device allocators

### CPU
`CpuAllocator` is a process-wide singleton (`CpuAllocator::get()`), reached
through `allocator_for(Device::Cpu(0))` which caches it in a `OnceLock`.
It wraps Rust's `std::alloc` with **64-byte alignment** to keep SIMD loads
happy, returns `alloc::handle_alloc_error` on failure, and treats `nbytes == 0`
specially by returning a dangling but aligned pointer (a zero-size `Layout` is
valid but of no use).

CPU is also one of the backends that does **not** use a `Region`: it
allocates exactly what is asked, directly. Its profiling works differently
from the static allocator's: it first checks
`crate::profiler::memory_enabled()` and, when a session is recording memory,
inserts the block's address into a `PROFILED` set and reports it. Frees then
report **only** blocks found in that set, matching PyTorch's
`ProfiledCPUMemoryReporter` — memory allocated before the session started is
not reported when it is freed. `PROFILED_COUNT` is checked before taking the
lock so the common case of freeing non-profiled memory never contends.

### Meta
`MetaAllocator::allocate` returns `NonNull::dangling()` and `deallocate` does
nothing. Meta tensors carry shape and dtype but no bytes; nothing ever
dereferences the pointer.

### CUDA
`cuda.rs` is just the FFI: `malloc(index, nbytes)` and `free(index, ptr)`
over `cudaMalloc`/`cudaFree`. Both call `CUDA.set_device(index)` first,
mirroring c10; without that, memory could be allocated on the wrong device or
freed from a thread with a different current device. Failure panics with
`"CUDA out of memory: failed to allocate {n} bytes on cuda:{i}"`. Everything
is behind `#[cfg(lumen_cuda_linked)]`. Each CUDA device's allocator is its
`CudaAllocator` from the registry, sized by `config::static_allocator_bytes()`
when created.

### MPS
`mps/mod.rs` is just the FFI: `malloc(nbytes)` and `free(ptr)` over
`lumen_mps_alloc` / `lumen_mps_free` (the shim). The device's allocator is its
`MpsAllocator`, whose region is a Shared-mode buffer, so it is
host-addressable — `tests.rs` writes and reads back through the raw pointer to
prove it.

The shim (`mps/mps_shim.mm`) does the Objective-C work:

* `lumen_default_device()` creates the system default `MTLDevice` exactly once
  via `dispatch_once` (the call is documented as expensive).
* `lumen_mps_alloc` makes a Shared-mode `MTLBuffer` and records it in a global
  `std::map<char *, id<MTLBuffer>>` keyed by the buffer's **contents**
  pointer. `lumen_mps_free` erases that entry, which releases the buffer
  under ARC.
* `lumen_mps_buffer` / `lumen_mps_buffer_pointer` do a `upper_bound` lookup so
  that a pointer *inside* a buffer — i.e. any individual allocation within a
  static allocator's region — resolves back to its owning buffer and byte
  offset. Ops use this to bind tensor memory, the equivalent of PyTorch's
  `getMTLBufferStorage`.
* `lumen_mps_limits` reports alignment, `vm_page_size`, `maxBufferLength` and
  `recommendedMaxWorkingSetSize` for size math on the Rust side.

### TPU
`tpu.rs`, over `lumen/device/tpu`'s PJRT shim. Each TPU's `TpuAllocator` is a
`Region` like CUDA's and MPS's, sized by `static_allocator_bytes`, its range
one **`u32[rows, 1, 128]` PJRT buffer**: rows of 512 bytes (`ROW`), tiled
`(1, 128)`, so stored byte for byte as the host does and addressable by a
program at any row (see [`../device/tpu/README.md`](../device/tpu/README.md)).
Every allocation is rounded up to whole rows, and its capacity down to a
multiple of `8 * 64 * 9 * 5 * 7` rows (~79 MiB): a SparseCore launch views
the whole buffer as rows of a table's width (`lumen/kernels/tpu`), so its
rows must be a multiple of that width's tile rows. TPU memory is not
host-addressable: the pointers handed out are the runtime's address of
that buffer plus the allocation's offset, never dereferenced.
`copy_to_host(index, ptr, dst)` reads an allocation back at its offset (the
TPU-to-host copy kernel's); `run(index, executable, rows, scalars, host)`
runs a TPU program (`lumen/kernels/tpu`) on the buffer, the rows of the TPU
pointers `rows` its first scalars (each must start a row), and keeps the
handle of its result, the same memory donated and written. The allocator
keeps its buffer handle itself (`Region::base` is the CUDA and MPS
allocators').

## Configuration

`config.rs` holds `static_allocator_bytes`, an `AtomicUsize` defaulting to
`1 << 30` (1 GiB) per device. It is read when a device's static allocator is
*created* — that is, on that device's first allocation. Changing it later
affects only devices that have not allocated yet; an existing allocator keeps
its size. `lumen.config.static_allocator_bytes` in Python reads and writes the
same atomic, and `lumen.config` also exposes `compiler` (the graph
compiler's flags, defined in `lumen/graph_compiler`).

## Usage across the crate

* `lumen/lib.rs` re-exports `Allocator`, `CpuAllocator` and `DataPtr` at the
  crate root.
* `lumen/tensor/storage.rs` is the main consumer: `Storage::new` calls
  `allocator_for` and panics on an unavailable device.
* `lumen/tensor/dlpack.rs` uses `DataPtr::from_raw_parts` to take ownership of
  externally allocated buffers with the producer's deleter as the allocator.
* `lumen/device/python.rs` and `lumen/stream/python.rs` call `allocator_for`
  purely as an **availability check** at the Python boundary, converting the
  error into a `RuntimeError`.
* The profiler receives allocation events through `Region` and
  `CpuAllocator`; `lumen/profiler/tests.rs` drives a `Region` on the
  CPU's memory to test them.

## Tests

`tests.rs` is split into these modules:

* `cpu` — singleton identity, 64-byte alignment, write/read round-trip,
  zero-byte safety, repeated alloc/free cycles.
* `region` — the meat, on a test allocator owning a `Region` over the CPU's
  memory, counting its reservations and releases: the region is reserved
  once and released with the allocator, `DataPtr` gives
  its allocation back on drop, allocations are aligned bumps, freed memory is
  reused while others are live, a loop with live inputs does not grow, freed
  neighbours merge, a full allocator refuses (and the panic names the
  setting), zero-byte allocations are fine, and the returned memory is
  actually usable.
* `devices` — cross-device behavior that runs anywhere: CPU always available,
  every available device has one shared allocator, `Storage` is `Send +
  Sync`, and unavailable devices are reported rather than panicking
  unexpectedly.
* `cuda` / `mps` / `tpu` — hardware-gated, with `require_cuda!` /
  `require_mps!` / `require_tpu!` macros that print "unavailable,
  skipping" and return early. The TPU tests run one at a time
  (`device::tpu::ONE_AT_A_TIME`, which `kernels/tpu`'s tests hold too: they
  measure the device's memory, which the others would change): the region is in the
  device's memory once reserved and gone when dropped, offsets are aligned
  and reused, reads at any offset agree, reads past the region fail, and a
  TPU tensor copies to the host.
