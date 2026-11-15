# horizontal

Horizontal fusion merges small, independent kernels of the same kind into one launch: a multi-output
`Fusion` whose outputs each come from one member. The `horizontal_fusion` slot is the last pass in the
pass list; MPS instead calls the same code from inside its `fusion` pass.

## Files

- `mod.rs` — the grouping algorithm (`fuse`), the `Target` trait, the `Kernels` target, `Grouping`, the
  `HorizontalFusion` pass, and helpers for codegen.
- `triton.rs` — Triton runs it with `fuse` and `horizontal_fusion`: candidates from
  `fusion::triton::horizontal_candidate` (non-trivial `elementwise` fusions without reductions or
  matmuls), `Grouping::Tiling`, at most 64 buffers.
- `tpu.rs` — TPU runs it with `horizontal_fusion`: candidates from `fusion::tpu::horizontal_candidate`
  (non-scalar `elementwise` fusions of TPU-fusible nodes not written in place into a concat),
  `Grouping::Shape`, at most 32 buffers.

MPS has no impl here (identity pass in this slot); its `fusion` pass calls `horizontal::fuse` with a
`Kernels` target of its own (`Grouping::Tiling`, 30 buffers) when `horizontal_fusion` is on. CPU and
CUDA get the identity pass.

## Grouping

`fuse(graph, scalars, target)` walks the graph and puts candidate nodes with equal keys into groups:

- Only nodes whose output has at most `MAX_ELEMENTS` (2^20) elements are candidates; larger kernels fill
  the device alone.
- A group holds at most 32 members and must fit the target's buffer limit; at most 4 groups per key stay
  open, the oldest closing when another opens.
- A node joins only when this keeps the graph acyclic (no path from members' readers back into the group)
  and moving reads later or writes earlier extends buffer lifetimes by at most 16 MiB.
- Member fusions are inlined into one body; a fusion's extra outputs (its `FusionOutput` readers) carry
  over. Nodes are then re-emitted so each group runs once all its reads are ready.

`Grouping::Tiling` keys kernels that stage transposes through tiles (`tiled`) by exact shape and puts
every other kernel in one key, limiting tiled groups to few staged transposes (`heroes_fit`) and checking
the merged body keeps the tiling (`keeps_tiling`). `Grouping::Shape` keys by output shape.

`HorizontalFusion` names every merged fusion with its `name` field (`elementwise` on Triton and TPU);
MPS's target names each by requesting a `Kernel::Fusion`.

## Used elsewhere

`segments(body)` groups a fused body's outputs by element count; a kernel runs each group on its own
range of the grid, so one launch can cover several sizes. The MPS, Triton and CUDA elementwise codegen,
`codegen::triton` and the MPS kernel runtime read it.
