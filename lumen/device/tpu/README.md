# `lumen/device/tpu`

Google TPUs, through **libtpu's PJRT C API**: the only way into a TPU from
outside JAX. This folder owns the process's one PJRT client, its devices
(`tpu:0`, `tpu:1`, ...), the device-memory calls the TPU allocator
([`../../allocator/tpu.rs`](../../allocator/README.md)) is built on, and
compiling and running the programs of lumen's TPU kernels
([`../../ops/tpu/`](../../ops/tpu/README.md)).

libtpu is **loaded at run time** (`dlopen`), never linked: the crate builds
anywhere, and a machine without libtpu or TPUs simply reports no TPU devices.
It is found at `TPU_LIBRARY_PATH` (the variable JAX and torch_xla read),
else by the loader's search for `libtpu.so`. A pip-installed libtpu lives at
`<venv>/lib/python3.*/site-packages/libtpu/libtpu.so`.

Built with the **`tpu` Cargo feature, on Linux**: `build.rs`
(`build_tpu_shim`) compiles `pjrt.c` and sets `cfg(lumen_tpu_linked)`.
Without it, the `Tpu` backend still exists and reports no devices.

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `Tpu` / `TPU`, the `DeviceBackend` (name `"tpu"`, device count, `check` with the shim's reason when none are found); the shim's FFI (`ffi`), `error()`, and for tests `memory(index)` and `ONE_AT_A_TIME` (the lock every TPU test holds). |
| `pjrt.c` | The C shim over the PJRT API: loading libtpu and creating the client once, the devices, reserving device memory (`lumen_tpu_alloc`, a `u32[rows, 1, 128]` buffer), freeing it, reading it back at an offset (`lumen_tpu_read`), compiling a StableHLO program for a device (`lumen_tpu_compile`; a portable executable launches at twice the cost), running one on the memory, donated (`lumen_tpu_run`), memory statistics, and libtpu's profiler (`lumen_tpu_profiler_start` / `_stop` / `_free`: a session's XSpace, read by `lumen/profiler/tpu.rs`). |
| `pjrt_c_api.h` | XLA's PJRT C API header (`xla/pjrt/c/pjrt_c_api.h`, Apache 2.0), vendored. |
| `profiler_c_api.h` | XLA's profiler plugin C API header (`xla/backends/profiler/plugin/profiler_c_api.h`, Apache 2.0), vendored: PJRT's profiler extension's functions. |

## How TPU memory works (and why the allocator is shaped this way)

What libtpu does (0.0.17 to 0.0.50, PJRT API 0.69 on), as probed on a v6e:

* **A `u32[rows, 1, 128]` buffer is byte-linear.** XLA tiles it in rows,
  `(1, 128)` (a dimension of 1 before the last two gets no sublane tiling),
  so its raw bytes on the device are the host's, byte for byte: the region a
  `Region` allocator needs, addressable by byte offset. A program can slice
  it at any row (512 bytes) and reshape rows to a matrix. (A `u8` buffer is
  not linear: TPUs pack sub-32-bit elements in tiles, `(1024)(128)(4,1)`,
  even when a linear layout is requested; a 2-D `u32[rows, 128]` one is, but
  tiled `(8, 128)`, so a program could only slice it every 8 rows, 4 KiB.)
  `lumen_tpu_alloc` reserves one with `PJRT_Client_CreateUninitializedBuffer`.
* **Memory is not host-addressable**, and there is no view of a sub-range:
  `PJRT_Client_CreateViewOfDeviceBuffer` is not implemented on TPU. A
  pointer the allocator hands out is the runtime's address of the region's
  buffer (`PJRT_Buffer_UnsafePointer`) plus the allocation's offset: a key,
  never dereferenced.
* **Reads at an offset work:** `PJRT_Buffer_CopyRawToHost`, which wants host
  memory and offsets aligned to 32 bytes for the TPU's DMA;
  `lumen_tpu_read` stages through an aligned buffer and widens the copy to
  aligned bounds, so any offset and length work.
* **Writes are programs.** PJRT has no "copy host bytes into an existing
  buffer" (its async transfer manager writes at offsets, but its buffer
  cannot be read until its last transfer). Every write is a program run on
  the region donated: `lumen_tpu_run` passes it as the last argument, which
  the program aliases to its one result, and hands back the result's handle
  (the same memory). A copy from the host uploads the bytes as an argument
  (`PJRT_Client_BufferFromHostBuffer`) for a kernel to copy in.
* **Programs** are StableHLO text (`format = "mlir"`) of one
  `tpu_custom_call` (a kernel in lumen's Pallas, lowered by lumen:
  `lumen/ops/tpu`),
  compiled once with
  hand-encoded `CompileOptionsProto` bytes: one replica, one partition,
  `compile_portable_executable`, so one executable runs on any device
  (`execute_device`).

## Invariants & gotchas

* **One client per process**, created on first use (`pthread_once`):
  `device_count()`, the first `check()` or allocation. Creating it claims the
  machine's TPUs; another process (a JAX program) cannot use them meanwhile.
* **The plugin's API may be older than the header.** Each argument struct
  carries this header's `struct_size` (a plugin accepts sizes at least its
  own); every function is checked present in the plugin's table (`HAS`)
  before it is used, and a missing one means no TPU devices, with the reason.
* **Errors are strings, not panics, until the allocator:** the shim records
  the last PJRT error (`lumen_tpu_error`); `check()` quotes it when no device
  is found, and the allocator's out-of-memory panic includes it.
* **Runs are asynchronous, reads are not:** `lumen_tpu_run` returns once the
  program is enqueued (its scalars and host bytes are freed then: the
  runtime keeps them until it has run); the next run takes its result, so
  runs are in order. Each read awaits its PJRT event, which waits for the
  runs before it, so a TPU's stream (`lumen/stream/tpu.rs`) has nothing to
  wait for.

## Related

- [`../`](../README.md) — the `DeviceBackend` trait and the `Device` enum (`Device::Tpu`).
- [`../../allocator/`](../../allocator/README.md) — `TpuAllocator`, a `Region` over the buffer this folder reserves.
- [`../../ops/device_transfer/`](../../ops/device_transfer/README.md) — copies to and from a TPU (`tpu.rs`).
- [`../../ops/tpu/`](../../ops/tpu/README.md) — the kernels: lumen's Pallas, wrapped, launched.
- [`../../stream/`](../../stream/README.md) — `TpuStream`.
