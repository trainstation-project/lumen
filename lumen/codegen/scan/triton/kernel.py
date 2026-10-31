"""Triton scan kernel: inclusive cumulative sums along one axis, a tile of lines a program, each line scanned
``LANES`` elements at a time with every chunk's total carried into the next.

The input ``in0`` and output ``out`` arrive in ``EXTRA``; ``LOAD0`` and ``STORE`` read and write them at flat
indices through their layouts."""

import triton
import triton.language as tl


@triton.jit
def add(a, b):
    """``a + b`` in ``a``'s dtype: the scan's combine."""
    return (a + b).to(a.dtype)


@triton.jit
def scan_kernel(
    EXTRA,
    LOAD0: tl.constexpr,
    STORE: tl.constexpr,
    N: tl.constexpr,
    LENGTH: tl.constexpr,
    INNER: tl.constexpr,
    LANES: tl.constexpr,
    ROWS: tl.constexpr,
    REVERSE: tl.constexpr,
    ACCUM_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Scans program ``BLOCK_ID``'s ``BLOCK_SIZE // LANES`` of the ``N`` lines, each ``LENGTH`` long with
    elements ``INNER`` apart, accumulating in ``ACCUM_DTYPE`` and walking the axis backwards when ``REVERSE``.
    With ``ROWS`` the tile's lines are its rows (the axis innermost in memory), else its columns."""
    in0, out = EXTRA[0], EXTRA[1]

    OUTPUTS: tl.constexpr = BLOCK_SIZE // LANES
    O = tl.program_id(0).to(tl.int64) * OUTPUTS + tl.arange(0, OUTPUTS)
    # Offset of each line's first element: outer index times the axis span plus the inner index.
    JO = (O // INNER) * (LENGTH * INNER) + O % INNER
    CARRY = tl.zeros([OUTPUTS], ACCUM_DTYPE)

    for START in range(0, LENGTH, LANES):
        K = START + tl.arange(0, LANES)

        if REVERSE:
            JK = (LENGTH - 1 - K) * INNER
        else:
            JK = K * INNER

        if ROWS:
            J = JO[:, None] + JK[None, :]
            MASK = (O < N)[:, None] & (K < LENGTH)[None, :]
        else:
            J = JK[:, None] + JO[None, :]
            MASK = (K < LENGTH)[:, None] & (O < N)[None, :]

        X = LOAD0(in0, J, MASK).to(ACCUM_DTYPE)
        X = tl.where(MASK, X, tl.zeros_like(X))

        if ROWS:
            STORE(out, J, CARRY[:, None] + tl.associative_scan(X, 1, add), MASK)
            CARRY = CARRY + tl.sum(X, 1, dtype=ACCUM_DTYPE)
        else:
            STORE(out, J, CARRY[None, :] + tl.associative_scan(X, 0, add), MASK)
            CARRY = CARRY + tl.sum(X, 0, dtype=ACCUM_DTYPE)
