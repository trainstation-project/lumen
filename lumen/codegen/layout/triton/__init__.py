"""Triton tiled transpose kernel: the fixed tile grid of a loop computed from transposes, its work a
generated ``BODY``.

``BODY(EXTRA, BLOCK_ID_Z, BLOCK_A, BLOCK_B, MASK)`` computes and stores the outputs on the tile of rows
``BLOCK_A`` (the merged dims contiguous in each transposed input) by columns ``BLOCK_B`` (the merged
innermost dims, contiguous in the output) of batch ``BLOCK_ID_Z``, where ``MASK`` holds, reading the
step's tensors and scalars from ``EXTRA`` (its inputs, then its outputs after the first, then the
first)."""

import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    EXTRA,
    BODY: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One ``TILE`` x ``TILE`` tile of an ``A`` x ``B`` output a program, the tiles of a batch
    consecutive, along ``B`` first; ``BLOCK_SIZE`` (a loop launch's) goes unread."""
    NUM_BLOCKS_A = (A + TILE - 1) // TILE
    NUM_BLOCKS_B = (B + TILE - 1) // TILE

    BLOCK_ID = tl.program_id(0).to(tl.int64)
    BLOCK_ID_Z = BLOCK_ID // (NUM_BLOCKS_A * NUM_BLOCKS_B)
    BLOCK_ID_A = BLOCK_ID // NUM_BLOCKS_B % NUM_BLOCKS_A
    BLOCK_ID_B = BLOCK_ID % NUM_BLOCKS_B

    BLOCK_A = BLOCK_ID_A * TILE + tl.arange(0, TILE)
    BLOCK_B = BLOCK_ID_B * TILE + tl.arange(0, TILE)
    MASK = (BLOCK_A[:, None] < A) & (BLOCK_B[None, :] < B)

    BODY(EXTRA, BLOCK_ID_Z, BLOCK_A, BLOCK_B, MASK)
