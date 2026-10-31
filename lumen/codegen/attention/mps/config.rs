//! Metal flash-attention block sizes, the largest whose tiles fit in threadgroup memory.

use crate::graph_compiler::common::attention::{Attention, Backward};

/// Query rows per block for the backward kernel, the largest that fits in threadgroup memory.
pub(crate) fn backward_block(b: &Backward) -> usize {
    let bytes = |n: usize| n * (b.h + b.hv) * b.dtype.size_of() + 2 * n * 4;

    [32, 16, 8]
        .into_iter()
        .find(|&n| bytes(n) <= 28 << 10)
        .unwrap_or(8)
}

/// Query rows per block when dQ is fused into the dK/dV kernel, or `None` if even 16 rows
/// don't fit in threadgroup memory alongside the key tile.
pub(crate) fn backward_block_with_dq(b: &Backward) -> Option<usize> {
    const KEYS: usize = 64;
    let t = b.dtype.size_of();
    let bytes = |n: usize| n * (b.h + b.hv) * t + 2 * n * 4 + KEYS * b.h * t + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

/// Whether the forward uses the decode kernel (few query rows).
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// Keys per block for the forward kernel, the largest whose K/V tiles fit in threadgroup memory.
pub(crate) fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    let bytes = |bk: usize| bk * (a.h + a.hv) * t;

    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}
