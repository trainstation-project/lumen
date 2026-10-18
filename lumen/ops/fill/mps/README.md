# `lumen/ops/fill/mps`

The MPS `fill_` kernel: compute shaders for dense and strided fills (modelled
on PyTorch's `fill_mps_kernel`), encoded into lumen's MPS stream without
waiting. Called from the `FILL` registry in [`../mod.rs`](../mod.rs)
(`DispatchKey::Mps`), i.e. from `Tensor::fill_` / `zeros` / `ones` on MPS
tensors. Not used by graph plans (their `full` is in
[`lumen/ops/factory/mps`](../../factory/mps/)).

## Files

| File | Contents |
| --- | --- |
| `mod.rs` | `fill<T>`: canonicalises the view, picks the kernel name for the profiler, calls `lumen_mps_fill`, handles its status. |
| `shim.mm` | `lumen_mps_fill` (C ABI): compiles the pipelines once, binds the buffer and dispatches. |
| `kernels.metal` | `fill_u8/u16/u32/u64` (contiguous) and `fill_strided_u8/u16/u32/u64`. |

## How it works

- Host (`mod.rs`): drop size-1 dims, sort by stride descending, and test
  whether the elements form one dense block. If so, `strides` is passed as
  null and a contiguous kernel runs (this also covers e.g. a fully
  transposed dense tensor); otherwise the strided kernel runs.
- Kernels are dtype-agnostic by element width (1/2/4/8 bytes); the value is
  passed as raw bytes.
- Contiguous (`fill_u*`): grid-stride stores of a vector type (`uchar4`,
  `ushort4`, `uint4`, `ulong2`), then thread 0 writes the tail. Grid is
  `min(256 * 32, roundup(count, 256))` threads, at least 256, threadgroups of
  `min(maxTotalThreadsPerThreadgroup, 256)`.
- Strided (`fill_strided_u*`): 2-D dispatch `(count / sizes[0], sizes[0])`;
  y indexes dim 0, x is decomposed over dims 1.. with dim 1 varying fastest.
- The buffer is bound at offset 0 and the start is passed in elements
  (`offset / elem_size`), so the tensor's start must be element-aligned.

## Status codes

`0` encoded; `-1` the pointer is not in a lumen MPS segment (nothing
submitted: `mps::cancel`, then the CPU kernel runs, since MPS memory is
unified); anything else (`-2`: no pipeline, no encoder, or a strided fill
over `UINT32_MAX` elements) cancels and panics.

## Invariants & gotchas

- The Metal source is a separate library: `build.rs` embeds
  `kernels.metal` alone as `kMpsFillSource` in `mps_fill_source.h`; it does
  not see `lumen/ops/mps/kernels.metal`, hence its own `#include
  <metal_stdlib>`.
- The extern block in `mod.rs` says "In mps.mm", but `lumen_mps_fill` is
  defined in `shim.mm`.
- `mps::submit` is called with the tensor kept alive and op name `"Fill"`
  for the profiler.

## Related

- [`lumen/ops/fill`](../) — `fill_op` and the CPU / CUDA kernels.
- `lumen/stream/mps` — `submit`, `cancel`, encoder and completion plumbing.
- `lumen/allocator/mps` — `lumen_mps_buffer`, which maps a pointer to its segment.
