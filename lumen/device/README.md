# device

Devices and the per-type backend queries: which device types exist (`cpu`, `mps`, `cuda`, `tpu`, `meta`), how many of each are present, and how to select one. Every tensor, allocator, plan and stream is tied to a `Device` from here.

## Files

- `mod.rs` holds the `Device` enum (a type plus an index, printed as `"type:index"`) and the `DeviceBackend` trait (`name`, `device_count`, `is_available`, `check`, `set_device`). `Device::backend()` maps a device to its static backend.
- `cpu.rs` and `meta.rs` are the host and meta backends. Meta tensors carry a type and shape but no data.
- `mps.rs` asks the Metal shim whether a Metal device exists.
- `cuda.rs` queries the CUDA runtime (`cudaGetDeviceCount`, `cudaSetDevice`) when lumen is linked against it; `set_device` makes a device current before work is launched on it.
- `tpu/` is the TPU backend:
  - `mod.rs` declares the FFI to the PJRT shim (open, allocate, read, compile, run, wait, ready events, profiler, memory stats) and the `Tpu` backend.
  - `pjrt.c` is the C shim over the PJRT C API. It `dlopen`s `libtpu.so` (or `TPU_LIBRARY_PATH`) once and keeps one client.
  - `pjrt_c_api.h` and `profiler_c_api.h` are the vendored PJRT headers.
  - `layout.rs` is the TPU-tiled layout (`Layout`): rank-2-and-up tensors are stored in tiles of `8 * pack` rows by 128 columns over their last two dims; rank 0 and 1 are padded to 512-byte rows. `tile`/`untile` convert host bytes to and from it, `bytes` and `alignment` size it, and `mosaic` names it for Mosaic kernels.
- `python.rs` binds the `lumen.device` class (modelled on `torch.device`, with an optional index) and the `lumen.cpu`, `lumen.mps`, `lumen.cuda`, `lumen.tpu` and `lumen.meta` modules (`device_count`, `is_available`, `synchronize`). `parse_device` reads a device argument; `resolve_device` also checks that an allocator exists for it.
- `__init__.py` and `_C.pyi` expose and type the Python side.
- `tests.rs` holds the unit tests.

## How it connects

- `lumen/allocator` picks an allocator per `Device` (`allocator_for`).
- The TPU layout is used by the allocator (`lumen/allocator/tpu.rs`), device transfers (`lumen/runtime/device_transfer/tpu`) and the Pallas kernels (`lumen/kernels/pallas`).

## Conventions

- TPUs do not store 64-bit dtypes; `Layout::new` fails for them. Bools are stored as int32.
- TPU tensors live TPU-tiled in the shared device region; anything reading or writing their bytes from the host must go through `Layout::tile`/`untile`.
- A device type that is not linked into the build reports zero devices rather than failing.
