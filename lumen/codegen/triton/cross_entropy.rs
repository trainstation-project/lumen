//! Triton linear cross entropy: a GEMM of the hidden states and the vocabulary's weight whose epilogue
//! reduces each output tile to its rows' max and sum of exponentials (and picks a row's label's logit
//! in the tile holding it), then a kernel combining a row's tiles into its `logsumexp` and loss. The
//! logits never reach memory.

use indoc::formatdoc;

use super::{at, load, named};

/// Generates the GEMM epilogue of a linear cross entropy over `rows` rows of `classes` logits, whose
/// labels have layout `target`: a tile of `BLOCK_SIZE_N` classes writes, per row, its max to `ptr` and
/// its sum of `exp(logit - max)` to `EXTRA[0]` at `row * tiles + tile`, and the label's logit to
/// `EXTRA[1]` from the tile holding it (a store masked to the label's column, no reduction).
pub(crate) fn cross_entropy_epilogue(
    rows: usize,
    classes: usize,
    target: (&[usize], &[usize]),
) -> (String, String) {
    let (label, source) = load(target.0, target.1);
    let labels = source.replace(&format!("def {label}("), "def LOAD_LABEL(");
    named(
        "cross_entropy_epilogue",
        formatdoc!(
            "
            {labels}

            @triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
                sums = EXTRA[0]
                picked = EXTRA[1]
                labels = EXTRA[2]
                width = value.shape[1]
                tiles = ({classes} + width - 1) // width
                at = m * tiles + tl.min(n, 1)[:, None] // width
                row_mask = m < {rows}
                x = tl.where(mask, value, float(\"-inf\"))
                peak = tl.max(x, 1)[:, None]
                total = tl.sum(tl.where(mask, tl.exp(x - peak), 0.0), 1)[:, None]
                tl.store(ptr + at, peak, mask=row_mask)
                tl.store(sums + at, total, mask=row_mask)
                hit = mask & (n == LOAD_LABEL(labels, m, row_mask).to(tl.int64))
                tl.store(picked + m + n * 0, value, mask=hit)
            "
        ),
    )
}

/// Generates the kernel combining a linear cross entropy's `tiles` partials a row (inputs: the tiles'
/// maxes, their sums of exponentials and the label's logit) into each row's loss and `logsumexp`, the
/// rows' `[2, n]` result: one row a program, `BLOCK_SIZE` tiles at a time.
pub(crate) fn cross_entropy_combine(tiles: usize, out: (&[usize], &[usize])) -> (String, String) {
    let lse = at("n + o", out.0, out.1);
    let loss = at("o", out.0, out.1);
    named(
        "cross_entropy_combine",
        formatdoc!(
            "
            @triton.jit
            def NAME(
                in0,
                in1,
                in2,
                out,
                n,
                LOAD0: tl.constexpr,
                LOAD1: tl.constexpr,
                LOAD2: tl.constexpr,
                BLOCK_SIZE: tl.constexpr,
            ):
                o = tl.program_id(0).to(tl.int64)
                peak = tl.full([], float(\"-inf\"), tl.float32)
                for start in range(0, {tiles}, BLOCK_SIZE):
                    c = start + tl.arange(0, BLOCK_SIZE)
                    mask = c < {tiles}
                    x = tl.where(mask, LOAD0(in0, o * {tiles} + c, mask), float(\"-inf\"))
                    peak = tl.maximum(peak, tl.max(x, 0))
                total = tl.zeros([], tl.float32)
                for start in range(0, {tiles}, BLOCK_SIZE):
                    c = start + tl.arange(0, BLOCK_SIZE)
                    mask = c < {tiles}
                    x = LOAD0(in0, o * {tiles} + c, mask)
                    s = LOAD1(in1, o * {tiles} + c, mask)
                    total += tl.sum(tl.where(mask, s * tl.exp(x - peak), 0.0), 0)
                lse = peak + tl.log(total)
                tl.store(out + {loss}, lse - LOAD2(in2, o, None))
                tl.store(out + {lse}, lse)
            "
        ),
    )
}
