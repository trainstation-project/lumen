"""Triton dynamic-update-slice kernel: the fixed grid over an update's elements, writing them a generated
``BODY``.

``BODY(EXTRA, BLOCK, MASK)`` writes the update elements at the flat indices ``BLOCK`` where ``MASK`` holds
into the output at their starts, reading the step's tensors and scalars from ``EXTRA`` (its inputs, then its
output)."""

import triton
import triton.language as tl


@triton.jit
def dynamic_update_slice_kernel(EXTRA, BODY: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Program ``BLOCK_ID``'s ``BLOCK_SIZE`` elements of the ``N``-element update (of each update, for a
    chain of them)."""
    BLOCK_ID = tl.program_id(0).to(tl.int64)
    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    BODY(EXTRA, BLOCK, BLOCK < N)
