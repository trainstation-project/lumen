# `lumen/allocator/mps`

The Metal (MPS) allocator: `MpsAllocator`, a static-allocator `Region`
(see [`../README.md`](../README.md)) over one Shared-mode placement `MTLHeap`, plus the
Objective-C++ shim that creates buffers and maps raw pointers back to them.
Compiled only with `cfg(lumen_mps_linked)`; `mps_shim.mm` is part of the
`lumen_mps_shim` library built by `build.rs`.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | FFI to the shim, `MpsAllocator` (implements `Allocator` over a `Region` on `Device::Mps(0)`, each allocation placed as a buffer of its own), `Scratch` (a zeroed buffer outside the region, `None` when Metal has no memory for it: the autotuner's), and `allocator()`, the shared instance. |
| `mps_shim.mm` | The default `MTLDevice`, buffer creation/release, the pointer→buffer map, and device limits. |

## How it works

* `allocator()` creates one `MpsAllocator` on first use, sized by
  `config::static_allocator_bytes()` at that moment.
* `allocate` calls `Region::take(nbytes, heap, stream::mps::synchronize)`:
  the first call reserves the whole region with `lumen_mps_alloc_heap`; on a
  full region it synchronizes the MPS stream once (queued work may still
  hold tensors) and retries. It then places a buffer of the allocation's own
  over its bytes (`lumen_mps_place`), which kernels bind it by;
  `deallocate` drops that buffer (`lumen_mps_unplace`) and is
  `Region::give`.
* Every buffer is **untracked**: Metal does not order kernels by the
  buffers they share; the stream does (`stream/mps`: a memory barrier
  before each dispatch, events between encoders and command buffers).
  One tracked buffer for everything made Metal order each kernel after the
  last as if every pair depended: `neg` of 4M bfloat16 elements, cold
  inputs, 51 µs a call against 39 µs untracked with barriers and 35 µs
  with a buffer per tensor in a heap (on an M4 Pro, outside lumen).
* Dropping the allocator frees the region's buffer via `lumen_mps_free`.

## Shim (`mps_shim.mm`)

* `lumen_default_device()` — `MTLCreateSystemDefaultDevice()` once
  (`dispatch_once`); `lumen_mps_available()` reports whether it exists
  (used by `lumen/device/mps.rs`).
* `lumen_mps_alloc(nbytes)` — an untracked Shared-mode `MTLBuffer`, stored
  in a mutex-guarded `std::map` keyed by its contents pointer (the map holds
  the strong reference); `lumen_mps_free` erases it, releasing it under ARC.
* `lumen_mps_alloc_heap(nbytes)` — an untracked Shared-mode placement
  `MTLHeap` of `nbytes` and a buffer over all of it, kept in a second map:
  its contents pointer is the region's base (a buffer placed at offset `o`
  starts at base + `o`). `lumen_mps_place(ptr, nbytes)` places a buffer at
  `ptr`'s offset in the heap containing it, in the first map;
  `lumen_mps_unplace` drops it.
* `lumen_mps_buffer(ptr, &offset)` — `upper_bound` lookup so any pointer
  *inside* a buffer resolves to that buffer and byte offset, a placed buffer
  before the heap's whole one; the kernel shims (`kernels/mps/shim.mm`) bind
  tensors this way.
  `lumen_mps_buffer_pointer` is the unretained C-pointer version, used by
  `lumen.Stream.buffer`.
* `lumen_mps_limits` — alignment, page size, `maxBufferLength`,
  `recommendedMaxWorkingSetSize`.

## Gotchas

* Shared storage means the region is host-addressable; CPU reads are only
  valid after `stream::mps::synchronize()`.
* All tensors live in one heap, so its size caps total MPS memory:
  raise `lumen.config.static_allocator_bytes` before the first MPS allocation.

## Related

* [`../README.md`](../README.md) — `Allocator`, `DataPtr`, `Region`.
* [`../../stream/mps`](../../stream/mps/README.md) — synchronization and kernel encoding.
* [`../../device`](../../device/README.md) — MPS availability.
