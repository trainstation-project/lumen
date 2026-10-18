# `lumen/allocator/mps`

The Metal (MPS) allocator: `MpsAllocator`, a static-allocator `Region`
(see [`../README.md`](../README.md)) over one Shared-mode `MTLBuffer`, plus the
Objective-C++ shim that creates buffers and maps raw pointers back to them.
Compiled only with `cfg(lumen_mps_linked)`; `mps_shim.mm` is part of the
`lumen_mps_shim` library built by `build.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | FFI to `lumen_mps_alloc` / `lumen_mps_free`, `MpsAllocator` (implements `Allocator` over a `Region` on `Device::Mps(0)`), and `allocator()`, the shared instance. |
| `mps_shim.mm` | The default `MTLDevice`, buffer creation/release, the pointer→buffer map, and device limits. |

## How it works

* `allocator()` creates one `MpsAllocator` on first use, sized by
  `config::static_allocator_bytes()` at that moment.
* `allocate` calls `Region::take(nbytes, malloc, stream::mps::synchronize)`:
  the first call reserves the whole region with `lumen_mps_alloc`; on a
  full region it synchronizes the MPS stream once (queued work may still
  hold tensors) and retries. `deallocate` is `Region::give`.
* Dropping the allocator frees the region's buffer via `lumen_mps_free`.

## Shim (`mps_shim.mm`)

* `lumen_default_device()` — `MTLCreateSystemDefaultDevice()` once
  (`dispatch_once`); `lumen_mps_available()` reports whether it exists
  (used by `lumen/device/mps.rs`).
* `lumen_mps_alloc(nbytes)` — a Shared-mode `MTLBuffer`, stored in a
  mutex-guarded `std::map` keyed by its contents pointer (the map holds the
  strong reference); `lumen_mps_free` erases it, releasing it under ARC.
* `lumen_mps_buffer(ptr, &offset)` — `upper_bound` lookup so any pointer
  *inside* a buffer resolves to that buffer and byte offset; the kernel shims
  (`ops/mps/shim.mm`, `ops/fill/mps/shim.mm`) bind tensors this way.
  `lumen_mps_buffer_pointer` is the unretained C-pointer version, used by
  `lumen.Stream.buffer`.
* `lumen_mps_limits` — alignment, page size, `maxBufferLength`,
  `recommendedMaxWorkingSetSize`.

## Gotchas

* Shared storage means the region is host-addressable; CPU reads are only
  valid after `stream::mps::synchronize()`.
* All tensors live in one buffer, so the buffer size caps total MPS memory:
  raise `lumen.config.static_allocator_bytes` before the first MPS allocation.

## Related

* [`../README.md`](../README.md) — `Allocator`, `DataPtr`, `Region`.
* [`../../stream/mps`](../../stream/mps/README.md) — synchronization and kernel encoding.
* [`../../device`](../../device/README.md) — MPS availability.
