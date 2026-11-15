# layout

Code generation for moving elements between layouts. This covers transposes fused with elementwise ops,
staged through tiles so reads and writes are both coalesced, and TPU reshapes of TPU-tiled tensors.
Other layout ops (reshape, broadcast, slice) are views or index arithmetic inside elementwise kernels.

## Files

- `mod.rs` — the `LayoutCodegen` trait and the transpose tiling that every backend and the fusion passes
  share. The trait has two methods: `transpose(body, by_value, options)` and `reshape(from, to)`.
- `mps/` — `mod.rs` and `transpose.rs`, the Metal tiled transpose.
- `triton/` — `mod.rs` and `kernel.py`, the Triton transpose template.
- `tpu/` — `mod.rs` and `kernel.py`, the Pallas reshape.

There is no CUDA layout code.

## Transpose tiling (`mod.rs`)

`transpose_tiling(body)` returns a `TransposeTiling`, or `None` if no transpose qualifies. It applies
when the output of an elementwise body (no GEMM, reduction or row reductions) comes from transposes
that swap a contiguous input axis with the output's innermost axis. The tiling holds:

- `heroes`: the tiled transposes, at most `MAX_HEROES` and all sharing one tiling.
- `a` and `b`: the merged dims, contiguous in the input and in the output.
- `batch`: the remaining dims, flattened into the grid's z axis.

One program (a threadgroup on Metal) handles one `TILE` x `TILE` (32 x 32) tile. A side shorter than
`MIN_TILED` is not tiled. These users decide what to run from `transpose_tiling` (keep them in
agreement):

- `codegen/mps.rs` and `codegen/triton.rs` choose the transpose kernel.
- `elementwise/triton` uses it to size the grid.
- `runtime::kernel::mps` uses it for the launch.
- `graph_compiler/passes/horizontal` fuses only members that are all tiled or all untiled.

## Backends

**MPS (`mps/transpose.rs`).** `LayoutCodegen for Mps` generates a Metal kernel that stages each hero
through threadgroup tiles. It writes outputs through the given views (offset, strides). `grid` gives
the launch grid. The step runs as a fusion through `runtime::kernel::mps::Fusion`. `reshape` fails
because MPS reshape steps run the fixed strided copy (`dynamic_slice::mps::Copy`).

**Triton (`triton/`).** `kernel.py` holds the fixed `transpose_kernel`. Each program takes one tile of
an `A` x `B` output, with each batch's tiles consecutive along `B`. The kernel calls a generated
`BODY(EXTRA, BLOCK_ID_Z, BLOCK_A, BLOCK_B, MASK)`. `LayoutCodegen for Triton` generates that `BODY`
for the step's tensor layouts (`Tensors`) when it launches, from `elementwise/triton`. The step's
template is `Template::Transpose { a, b }`, run by the `Loop` kernel. `reshape` fails because a Triton
reshape is a view the elementwise kernels read through.

**TPU (`tpu/`).** `LayoutCodegen for Tpu` handles only reshapes, because transposes fuse into the TPU
elementwise kernels. A reshape is supported only when it splits a last dimension `a * w` into `[a, w]`
or merges them back. Any other reshape that moves elements between tiles fails. `kernel(step)` wraps it
as module `lumen.codegen.layout.tpu.kernel` with `Rows::Step`. `codegen/tpu.rs` uses that for
`Reshape` steps, and the `Pallas` kernel launches it. `kernel.py::program(spec)` copies `bm`-row blocks
between the flat and tiled forms.

None of these backends defines a Kernel struct or configs for this family.
