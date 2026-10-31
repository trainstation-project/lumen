"""Triton scatter-add kernel: a fixed kernel over the positions off the scattered axis, its tensors
contiguous (scatter-adds read no strided inputs)."""

import triton
import triton.language as tl


@triton.jit
def scatter_add_kernel(
    EXTRA,
    N: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adds each of the ``PICKS`` slices of the updates into ``out`` (already holding the operand) at the
    positions along the ``LENGTH``-long axis listed in the indices, for program ``BLOCK_ID``'s
    ``BLOCK_SIZE`` of the ``N`` ``(outer, inner)`` positions off the axis (``AFTER`` inner ones per outer).
    ``EXTRA`` holds the operand, indices, updates and ``out``; each program owns distinct positions, so the
    loop over picks is race-free."""
    indices, updates, out = EXTRA[1], EXTRA[2], EXTRA[3]

    BLOCK_ID = tl.program_id(0).to(tl.int64)
    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = BLOCK < N

    OUTER = BLOCK // AFTER
    INNER = BLOCK % AFTER

    for PICK in range(0, PICKS):
        AT = (OUTER * LENGTH + tl.load(indices + PICK).to(tl.int64)) * AFTER + INNER
        VALUE = tl.load(out + AT, mask=MASK) + tl.load(updates + (OUTER * PICKS + PICK) * AFTER + INNER, mask=MASK)
        tl.store(out + AT, VALUE, mask=MASK)
