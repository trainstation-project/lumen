//! Query and key block sizes of the Pallas flash-attention kernels.

/// TPU lane width; attention block sizes are multiples of it.
const LANES: usize = 128;

/// Largest query block.
const QUERIES: usize = 1024;

/// Largest key block.
const KEYS: usize = 4096;

/// Largest multiple of `LANES` up to `cap` that divides `n`, or `n` itself when none does.
fn block(n: usize, cap: usize) -> usize {
    (1..=cap / LANES)
        .rev()
        .map(|k| k * LANES)
        .find(|&b| n.is_multiple_of(b))
        .unwrap_or(n)
}

/// Query and key block sizes for sequence lengths `sq` and `sk`.
pub(crate) fn blocks(sq: usize, sk: usize) -> (usize, usize) {
    (block(sq, QUERIES), block(sk, KEYS))
}
