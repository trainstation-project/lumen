# canonicalize

`CanonicalizeDots` (pass name `canonicalize_dots`) puts in an explicit `Transpose` before each dot operand
a backend's matmul kernel can't read as laid out, and rewrites the dot's dimension numbers to match. It
runs after `split_k` and before the diamonds passes, so later passes and codegen see only operands the
kernel reads directly.

## Files

- `mod.rs` — the shared pass, `Readable`, `MatmulOrder` and `matmul_order`.
- `mps.rs` — MPS runs it always, with `reads_contiguous`: an operand is readable when the GEMM kernels'
  `reads_strided` accepts its contiguous strides (each batch, free and contracting dimension group merges
  into one stride). Built only when MPS is linked.
- `tpu.rs` — TPU runs it always, with its own rule: batch dims lead the operand; the lhs is batch, free
  dims, then one contracting dim (or batch, one contracting, one free); the rhs is batch then one
  contracting and one free dim in either order.

CPU, CUDA and Triton get the identity pass.

## Rewrite

- `CanonicalizeDots { readable }` holds the backend's rule. `Readable` is
  `fn(&Primitive, [&TensorType; 2], operand index) -> bool` (0 = lhs, 1 = rhs).
- `matmul_order(dot, lhs_rank, rhs_rank)` gives the permutations that put the lhs in
  batch, free (M), contracting (K) order and the rhs in batch, contracting (K), free (N) order, plus where
  each group ends (`lhs_split`, `rhs_split`).
- An unreadable lhs is transposed by `order.lhs`; its batch dims become `0..i` and its contracting dims the
  tail. An unreadable rhs is transposed by `order.rhs`; its batch dims become `0..i` and contracting dims
  `i..j`. A readable operand is left alone, so only the operands that need it are copied.
- Dtypes are unchanged; every other node is copied with its label.

`matmul_order` is also used by MPS's GEMM codegen (`codegen::dot_general::mps`) to read operands in the
same `[batch, M, K]` / `[batch, K, N]` form. Tests construct the pass directly with `mps::reads_contiguous`.
