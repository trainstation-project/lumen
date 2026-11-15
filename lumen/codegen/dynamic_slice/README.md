# dynamic_slice

Code generation and kernels for `dynamic_slice` and `dynamic_update_slice`: one slice or update at runtime start
indices, and an `updates` fusion (`crate::graph::UPDATES`), a chain of updates at constant starts written into
one operand. Codegen picks these kernels for the matching plan steps on MPS, Triton and TPU. CUDA has none.

## Files

- `mod.rs` defines the `DynamicSliceCodegen` trait. `updates(body, options)` generates the code for an `updates`
  fusion body: input 0 is the operand and inputs `1..` are the updates, which all have the same shape.
  `update(operand, update, by_value, options)` generates one update at runtime starts; the start inputs flagged
  in `by_value` are passed as scalars. A backend that runs a fixed kernel for an op returns an error from that
  method.
- `mps/`: Metal (built only with `lumen_mps_linked`).
  - `mod.rs`:
    - `updates` generates a kernel named with the `UPDATES` prefix. Thread `t` writes element `t` of every
      update into `out`, which already holds the operand. `update` returns an error because a single update
      runs the fixed kernel.
    - Kernel structs (no config, never tuned): `DynamicSlice` runs a lone `dynamic_slice` or
      `dynamic_update_slice` (`codegen/mps.rs` picks it), and `Copy` runs a `Reshape` step as a whole-buffer
      byte copy.
    - The free function `copy(src, dst, ty, keep, name)` launches `dynamic_slice_copy_<bytes>`. The plan runner,
      tensor copies, host-to-device transfers and the scatter-add also use it.
    - `MAX_RANK` (8) is the highest operand rank the kernels take.
  - `kernel.metal`: the fixed kernels `dynamic_slice_*`, `dynamic_update_slice_*` and `dynamic_slice_copy_*`.
    Slice and update kernels are instantiated per element byte width, per start dtype (i32/i64) and per 32- or
    64-bit offsets. Each thread handles 16 bytes of consecutive elements. `build.rs` compiles this file into
    the primitive-kernel library.
- `triton/`
  - `mod.rs` implements the trait for Triton, with the tensors' layouts as `Options`. Both methods return the
    `(name, source)` of a `BODY(EXTRA, BLOCK, MASK)` function, which loads and stores through helpers generated
    for those layouts. The `BODY` is generated when the step launches.
  - `kernel.py` holds the fixed `dynamic_update_slice_kernel`: a 1-D grid of `BLOCK_SIZE` blocks over the `N`
    update elements that calls `BODY`. It runs as `Template::DynamicUpdateSlice` in `codegen/triton.rs`, under
    the `Loop` kernel.
- `tpu/`: Pallas (built only with `lumen_tpu_linked`).
  - `mod.rs` implements the trait for TPU, producing a `Program`: the kernel name, a JSON spec, and the
    constant starts passed as `i32` scalars. `kernel(step)` returns the step's `codegen::tpu::Kernel` (module
    `lumen.codegen.dynamic_slice.tpu.kernel`, `Rows::Step`), or `None` for an empty update. The generic
    `Pallas` kernel in `codegen/tpu.rs` launches it.
    - The spec picks block sizes under fixed budgets (`STAGED`, `STREAMED`, `COPY_BYTES`).
    - It records whether each update is tile-aligned on its last two dims.
    - It records whether the region all updates touch can stay resident in VMEM (`RESIDENT`). When the
      resident updates cover whole rows, only the rows outside them are copied.
  - `kernel.py`: `program(spec)` builds the Pallas program. Aligned updates are DMA'd directly in
    double-buffered pieces. Unaligned updates are rolled into place inside tile-aligned windows. The kernel
    copies the operand into the output first unless the update is in place.

## Conventions

- Start indices are clamped to `[0, dim - size]`, so the block always stays inside the operand. This is the
  semantics of the op, not a safety check. The Metal and Triton kernels clamp at run time; for constant starts
  of an `updates` fusion, the TPU code clamps when it builds the spec.
- In an `updates` chain where updates overlap, later updates win. The MPS and Triton generators skip elements
  that a later overlapping update will overwrite. On TPU, overlapping windows are applied in order rather than
  in flight.
- An out-of-place update copies its operand into the output first. On MPS this happens when the output buffer
  is not the operand's.
