# allocator

Device memory for tensors. Every `Storage` (see `lumen/tensor`) owns a `DataPtr` taken from the allocator of its device; the plan (`lumen/plan`) and runtime (`lumen/runtime`) get their inputs, outputs and shared workspace from the same allocators.

## Files

- `mod.rs` holds the `Allocator` trait and `allocator_for(device)`.
- `region.rs` holds `Region`, the first-fit sub-allocator the accelerator backends share.
- `data_ptr.rs` holds `DataPtr`, the owned buffer that gives its bytes back to its allocator when dropped.
- `config.rs` holds the process-wide size of each accelerator's static region (`static_allocator_bytes`, default 1 GiB), `ALIGNMENT`, the alignment and granularity of every region block (256 bytes), and `PINNED_ALLOCATOR_BYTES`, the CUDA pinned pool's size (256 MiB). Python reads and sets it through `lumen.config` (`lumen/config/python.rs`).
- `cpu.rs` is `CpuAllocator`: host heap memory at the requested alignment (at least 64 bytes), reporting allocations to the memory profiler while profiling is on.
- `meta.rs` is `MetaAllocator`: dangling pointers for meta tensors, which never touch memory.
- `cuda.rs` is `CudaAllocator`: one `cudaMalloc`ed region per device, carved by `Region`. It also holds blocks taken while a CUDA graph captures (`hold`, `held`, `release`, used by `lumen/runtime/stream/cuda.rs`): a freed held block only returns to the region once its graph is dropped, because the graph still launches into it.
- `mps/mod.rs` is `MpsAllocator`: one Metal heap per process, reserved on first use, with a placed `MTLBuffer` registered for each block. `Scratch` is a separate zeroed Metal buffer outside the region, used by MPS plan steps that need temporary memory.
- `mps/mps_shim.mm` is the Objective-C side: it allocates heaps and buffers and maps raw pointers back to the `MTLBuffer` that contains them.
- `pinned.rs` (CUDA only) is `PinnedAllocator`: host memory from one `cudaHostAlloc`ed range carved by `Region`, for asynchronous copies from the device (a copy into pageable memory blocks until it finishes); a block that doesn't fit comes from the host heap. `pinned()` returns it, `is_pinned(ptr)` tells its blocks apart.
- `tpu.rs` is `TpuAllocator`: one PJRT runtime buffer per device holds every tensor, addressed in 512-byte rows (`ROW`). It also runs executables on that buffer (`run`), converts pointers to row indices (`rows_of`), waits on pending work and copies to the host.
- `tests.rs` holds unit tests for every backend; device tests return early when the device is missing.

## Key types

- `Allocator` (`allocate(nbytes, alignment)`, `deallocate(ptr, nbytes, alignment)`) is shared per device as `Arc<dyn Allocator>`; every allocator honors the alignment it is asked for (`Tensor` asks for `ALIGNMENT`, or a TPU layout's); `allocator_for(Device)` returns it or an error when the device is unavailable.
- `DataPtr` records its size and alignment and gives its bytes back to its allocator when dropped.
- `Region` reserves one contiguous range lazily, hands out blocks first-fit (`take`, or `try_take` without panicking) at the requested alignment, at least `ALIGNMENT` (256 bytes), and coalesces freed neighbours. When nothing fits it calls the backend's `synchronize` and retries once, then panics with a message pointing at `static_allocator_bytes`.

## Conventions

- Accelerator memory is one fixed static region per device; there is no growth. Changing `static_allocator_bytes` only affects allocators created afterwards.
- On MPS and TPU a tensor pointer is an offset into one device buffer; kernels address tensors through that shared buffer, so pointers must come from this region.
- TPU allocations are whole rows and the region size is rounded to a row granularity that common tile heights divide (see `lumen/device/tpu` for the tiled layout).
- `deallocate` is `unsafe`: the pointer must come from the same allocator at the same size and alignment, unused after the call.
