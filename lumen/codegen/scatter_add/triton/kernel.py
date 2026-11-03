"""Triton scatter-add kernels, over the operand viewed as ``[outer, LENGTH, AFTER]`` and the updates as
``[outer, PICKS, AFTER]`` (contiguous: scatter-adds read no strided inputs). By default one kernel adds each
update element atomically (``scatter_add_atomic_kernel``). Deterministic, as MPS's: a counting sort of the
picks by position (``count``, ``starts``, ``place``, ``sort``), then a program a block of a position's row
adding its updates in pick order (``rows``), so duplicates sum as the CPU sums them.

Every kernel takes the same constexprs: ``N``, the launch's count (update elements, picks, positions or rows
programs), and the sizes ``OUTER``, ``LENGTH``, ``AFTER`` and ``PICKS``."""

import triton
import triton.language as tl


@triton.jit
def scatter_add_atomic_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adds program ``BLOCK_ID``'s ``BLOCK_SIZE`` of the ``N`` update elements into ``out`` (already
    holding the operand), each at the position along the ``LENGTH``-long axis its pick's index lists, by an
    atomic add: updates at one position sum in an arbitrary order, as torch's ``index_add``. ``EXTRA``
    holds the operand, indices, updates and ``out``."""
    indices, updates, out = EXTRA[1], EXTRA[2], EXTRA[3]

    BLOCK_ID = tl.program_id(0).to(tl.int64)
    BLOCK = BLOCK_ID * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = BLOCK < N

    # Update element BLOCK is updates[AT_OUTER, PICK, AT_INNER]; it adds into out[AT_OUTER, indices[PICK],
    # AT_INNER].
    AT_INNER = BLOCK % AFTER
    PICK = (BLOCK // AFTER) % PICKS
    AT_OUTER = BLOCK // (AFTER * PICKS)

    AT = (AT_OUTER * LENGTH + tl.load(indices + PICK, mask=MASK).to(tl.int64)) * AFTER + AT_INNER
    tl.atomic_add(out + AT, tl.load(updates + BLOCK, mask=MASK), mask=MASK, sem="relaxed")


@triton.jit
def scatter_add_count_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Counts program ``BLOCK_ID``'s ``BLOCK_SIZE`` picks at their positions: ``EXTRA`` holds the indices
    and the (zeroed) int32 counts, a position each. Integer atomics, so the counts don't depend on order."""
    indices, counts = EXTRA[0], EXTRA[1]

    PICK = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = PICK < PICKS

    AT = tl.load(indices + PICK, mask=MASK).to(tl.int64)
    tl.atomic_add(counts + AT, tl.full([BLOCK_SIZE], 1, tl.int32), mask=MASK, sem="relaxed")


@triton.jit
def scatter_add_starts_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program: each position's segment start, ``starts[k] = sum(counts[:k])`` for ``k`` up to
    ``LENGTH`` (an exclusive prefix sum, ``BLOCK_SIZE`` counts at a time), zeroing the counts for the place
    kernel's cursors. ``EXTRA`` holds the counts and ``starts``."""
    counts, starts = EXTRA[0], EXTRA[1]

    RUN = tl.full([], 0, tl.int32)

    for FIRST in range(0, LENGTH, BLOCK_SIZE):
        AT = FIRST + tl.arange(0, BLOCK_SIZE)
        MASK = AT < LENGTH
        COUNT = tl.load(counts + AT, mask=MASK, other=0)
        tl.store(starts + AT, RUN + tl.cumsum(COUNT, 0) - COUNT, mask=MASK)
        tl.store(counts + AT, tl.zeros_like(COUNT), mask=MASK)
        RUN += tl.sum(COUNT, 0)

    tl.store(starts + LENGTH, RUN)


@triton.jit
def scatter_add_place_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Writes program ``BLOCK_ID``'s ``BLOCK_SIZE`` picks into their positions' segments of ``order``, at
    each segment's start plus its position's cursor; a segment's order is arbitrary until the sort kernel.
    ``EXTRA`` holds the indices, the starts, the (zeroed) cursors and ``order``."""
    indices, starts, cursors, order = EXTRA[0], EXTRA[1], EXTRA[2], EXTRA[3]

    PICK = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = PICK < PICKS

    AT = tl.load(indices + PICK, mask=MASK).to(tl.int64)
    SLOT = tl.atomic_add(cursors + AT, tl.full([BLOCK_SIZE], 1, tl.int32), mask=MASK, sem="relaxed")
    tl.store(order + tl.load(starts + AT, mask=MASK) + SLOT, PICK.to(tl.int32), mask=MASK)


@triton.jit
def scatter_add_sort_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Sorts position ``program_id(0)``'s segment of ``order`` ascending, so its updates add in pick order:
    up to ``BLOCK_SIZE`` picks in registers; a longer one rewritten by walking every index in order,
    ``BLOCK_SIZE`` at a time, keeping the position's picks in that order (a stable compaction). ``EXTRA``
    holds the indices, the starts and ``order``."""
    indices, starts, order = EXTRA[0], EXTRA[1], EXTRA[2]

    POSITION = tl.program_id(0).to(tl.int64)
    START = tl.load(starts + POSITION)
    SIZE = tl.load(starts + POSITION + 1) - START
    LANE = tl.arange(0, BLOCK_SIZE)

    if SIZE > BLOCK_SIZE:
        WRITTEN = tl.full([], 0, tl.int32)

        for FIRST in range(0, PICKS, BLOCK_SIZE):
            PICK = FIRST + LANE
            HIT = (tl.load(indices + PICK, mask=PICK < PICKS, other=-1).to(tl.int64) == POSITION).to(tl.int32)
            # Hits keep their order: each goes after the earlier hits of its block and of the blocks before.
            tl.store(order + START + WRITTEN + tl.cumsum(HIT, 0) - HIT, PICK, mask=HIT == 1)
            WRITTEN += tl.sum(HIT, 0)
    elif SIZE > 1:
        # Padded with PICKS, past every pick, the segment sorts to the front.
        PICKED = tl.load(order + START + LANE, mask=LANE < SIZE, other=PICKS)
        tl.store(order + START + LANE, tl.sort(PICKED), mask=LANE < SIZE)


@triton.jit
def scatter_add_rows_kernel(
    EXTRA,
    N: tl.constexpr,
    OUTER: tl.constexpr,
    LENGTH: tl.constexpr,
    AFTER: tl.constexpr,
    PICKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adds into program ``(POSITION, block)``'s ``BLOCK_SIZE`` of the ``OUTER * AFTER`` entries of ``out``
    at ``POSITION`` along the axis (already holding the operand) the updates its position's sorted segment of
    ``order`` lists, in pick order, each add rounded in the dtype as the CPU rounds it. ``EXTRA`` holds the
    updates, the starts, ``order`` and ``out``."""
    updates, starts, order, out = EXTRA[0], EXTRA[1], EXTRA[2], EXTRA[3]

    BLOCKS: tl.constexpr = (OUTER * AFTER + BLOCK_SIZE - 1) // BLOCK_SIZE
    PROGRAM = tl.program_id(0).to(tl.int64)
    POSITION = PROGRAM // BLOCKS
    # The entries off the axis this program adds into: out[AT_OUTER, POSITION, AT_INNER].
    ENTRY = (PROGRAM % BLOCKS) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    MASK = ENTRY < OUTER * AFTER
    AT_OUTER = ENTRY // AFTER
    AT_INNER = ENTRY % AFTER

    START = tl.load(starts + POSITION)
    SIZE = tl.load(starts + POSITION + 1) - START

    if SIZE > 0:
        AT = out + (AT_OUTER * LENGTH + POSITION) * AFTER + AT_INNER
        # Pick p's update at these entries is ROW + p * AFTER.
        ROW = updates + AT_OUTER * PICKS * AFTER + AT_INNER
        ACC = tl.load(AT, mask=MASK)

        # Four picks a step, their loads issued before the adds so they overlap; past the segment's end a
        # pick repeats its last one and its add is skipped.
        for I in range(0, SIZE, 4):
            P0 = tl.load(order + START + I).to(tl.int64)
            P1 = tl.load(order + START + tl.minimum(I + 1, SIZE - 1)).to(tl.int64)
            P2 = tl.load(order + START + tl.minimum(I + 2, SIZE - 1)).to(tl.int64)
            P3 = tl.load(order + START + tl.minimum(I + 3, SIZE - 1)).to(tl.int64)
            U0 = tl.load(ROW + P0 * AFTER, mask=MASK)
            U1 = tl.load(ROW + P1 * AFTER, mask=MASK)
            U2 = tl.load(ROW + P2 * AFTER, mask=MASK)
            U3 = tl.load(ROW + P3 * AFTER, mask=MASK)
            ACC = (ACC + U0).to(ACC.dtype)
            ACC = tl.where(I + 1 < SIZE, (ACC + U1).to(ACC.dtype), ACC)
            ACC = tl.where(I + 2 < SIZE, (ACC + U2).to(ACC.dtype), ACC)
            ACC = tl.where(I + 3 < SIZE, (ACC + U3).to(ACC.dtype), ACC)

        tl.store(AT, ACC, mask=MASK)
