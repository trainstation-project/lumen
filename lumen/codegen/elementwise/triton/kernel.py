"""Triton elementwise kernel: the fixed grid of an elementwise fusion, its elementwise work a generated ``BODY``.

``BODY(EXTRA, BLOCK, MASK, SEGMENT)`` computes and stores every output of grid segment ``SEGMENT`` at the
flat indices ``BLOCK`` where ``MASK`` holds, reading the step's tensors and scalars from ``EXTRA`` (its
inputs, then its outputs after the first, then the first)."""

import triton
import triton.language as tl


@triton.jit
def elementwise_kernel(EXTRA, BODY: tl.constexpr, SEGMENTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Program ``BLOCK_ID``'s ``BLOCK_SIZE`` elements: each segment's element count in ``SEGMENTS`` runs on
    its own consecutive range of programs, the first segment's first."""
    BLOCK_ID = tl.program_id(0).to(tl.int64)
    FIRST_BLOCK_ID = 0

    for SEGMENT in tl.static_range(len(SEGMENTS)):
        NUM_BLOCKS = (SEGMENTS[SEGMENT] + BLOCK_SIZE - 1) // BLOCK_SIZE

        if (BLOCK_ID >= FIRST_BLOCK_ID) & (BLOCK_ID < FIRST_BLOCK_ID + NUM_BLOCKS):
            BLOCK = (BLOCK_ID - FIRST_BLOCK_ID) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            BODY(EXTRA, BLOCK, BLOCK < SEGMENTS[SEGMENT], SEGMENT)

        FIRST_BLOCK_ID += NUM_BLOCKS
