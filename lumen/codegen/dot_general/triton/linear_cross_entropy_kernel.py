"""Triton linear cross entropy: a GEMM of the hidden states and the vocabulary's weight whose epilogue reduces
each output tile to its rows' max and sum of exponentials (and picks a row's label's logit in the tile holding
it), then a kernel combining a row's tiles into its ``logsumexp`` and loss. The logits never reach memory."""

import triton
import triton.language as tl


@triton.jit
def cross_entropy_epilogue(ptr, EXTRA, l, m, n, value, mask):
    """The GEMM epilogue: a tile of ``value.shape[1]`` classes writes, per row ``m``, its max to ``ptr`` and its
    sum of ``exp(logit - max)`` to ``sums`` at ``m * tiles + tile``, and the label's logit to ``picked`` from the
    tile holding it (a store masked to the label's column, no reduction). ``EXTRA`` holds ``sums``, ``picked``,
    the labels, then the rows, the classes and the labels' stride."""
    sums, picked, labels = EXTRA[0], EXTRA[1], EXTRA[2]
    rows, classes, stride = EXTRA[3], EXTRA[4], EXTRA[5]

    width = value.shape[1]
    tiles = (classes + width - 1) // width
    at = m * tiles + tl.min(n, 1)[:, None] // width
    row_mask = m < rows

    x = tl.where(mask, value, float("-inf"))
    peak = tl.max(x, 1)[:, None]
    total = tl.sum(tl.where(mask, tl.exp(x - peak), 0.0), 1)[:, None]
    tl.store(ptr + at, peak, mask=row_mask)
    tl.store(sums + at, total, mask=row_mask)

    hit = mask & (n == tl.load(labels + m * stride, mask=row_mask).to(tl.int64))
    tl.store(picked + m + tl.zeros_like(n), value, mask=hit)


@triton.jit
def cross_entropy_combine_kernel(EXTRA, N: tl.constexpr, TILES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Combines row ``program_id(0)``'s ``TILES`` partials (the tiles' maxes and sums of exponentials, and the
    label's logit) into its loss and ``logsumexp``, ``BLOCK_SIZE`` tiles at a time, written to the ``[2, N]``
    ``out``: the losses, then the ``logsumexp``s. ``EXTRA`` holds the maxes, the sums, the label logits and
    ``out``, all contiguous."""
    peaks, sums, picked, out = EXTRA[0], EXTRA[1], EXTRA[2], EXTRA[3]
    o = tl.program_id(0).to(tl.int64)

    peak = tl.full([], float("-inf"), tl.float32)
    for start in range(0, TILES, BLOCK_SIZE):
        c = start + tl.arange(0, BLOCK_SIZE)
        mask = c < TILES
        x = tl.load(peaks + o * TILES + c, mask=mask, other=float("-inf"))
        peak = tl.maximum(peak, tl.max(x, 0))

    total = tl.zeros([], tl.float32)
    for start in range(0, TILES, BLOCK_SIZE):
        c = start + tl.arange(0, BLOCK_SIZE)
        mask = c < TILES
        x = tl.load(peaks + o * TILES + c, mask=mask)
        s = tl.load(sums + o * TILES + c, mask=mask)
        total += tl.sum(tl.where(mask, s * tl.exp(x - peak), 0.0), 0)

    lse = peak + tl.log(total)
    tl.store(out + o, lse - tl.load(picked + o))
    tl.store(out + N + o, lse)
