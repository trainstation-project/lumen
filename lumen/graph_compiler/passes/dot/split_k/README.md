# split_k

`SplitK` (pass name `split_k`) splits a matmul with a long contraction but too few output tiles to fill
the GPU into partial products over K chunks, summed over the chunks. It runs after `linear_reordering`
and before `canonicalize_dots`, behind the `split_k` config flag.

## Files

- `mod.rs` — the shared rewrite `split_k(graph, &Rules)`, the `Rules` a backend supplies, the
  `divides` helper, and `SplitK`, the slot type the pipeline picks per device.
- `mps.rs` — MPS rules: tiles counted with the GEMM's `float_groups`, a target of 256 threadgroups, GEMV
  and skinny products of at most 4 rows kept whole, and an MLX-style preferred chunk count for small
  outputs with long K. Exposes `mps::split_k(graph)`, which tests call. Built only when MPS is linked.
- `triton.rs` — Triton rules: tiles of the untuned GEMM config (`CONFIGS[UNTUNED]`), a target of 128
  tiles, matrix-vector products (`dot_general::triton::gemv`, MPS's rule) kept whole for the GEMV kernel, no
  preferred count. Partials are always summed by a reduction, not atomics,
  so results stay deterministic.

CPU, CUDA and TPU get the identity pass. Both MPS and Triton run it only when `config.split_k` is on.

## Rewrite

A dot is split when it is unbatched, has one contracting dim per side, accumulates in a float dtype,
neither operand is a `Slice` output, and the backend's `unsplit` rule does not keep it whole.

- The chunk count is a power of two from `target / tiles(m, n)`, capped at 8, and only when the output has
  fewer than half the target tiles and K is at least 2048. A backend's preferred count (`Rules::chunks`)
  wins when larger.
- `divides` halves the count until every chunk has at least 256 elements and is a multiple of 8 elements
  (whole 16-byte vectors of 2-byte elements); fewer than 2 chunks means no split.
- Each operand's contracting dim `d` is reshaped to `[s, k / s]`; the dot then batches over the chunk
  index and contracts the chunk length, writing the accumulation dtype. A `ReduceSum` over axis 0 sums the
  chunks, and a `Cast` follows when the original output dtype differs from the accumulator.
- The new nodes are labelled `dot_general (split-K)` and `reduce_sum (split-K)`.
