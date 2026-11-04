//! Metal flash-attention block sizes, the largest whose tiles fit in threadgroup memory.

use crate::DType;
use crate::graph_compiler::common::attention::{Attention, Backward, grid_batch};

/// Query rows per block for the backward kernel, the largest that fits in threadgroup memory.
pub(crate) fn backward_block(b: &Backward) -> usize {
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |n: usize| n * ((b.h + b.hv) * b.dtype.size_of() + 32) + 2 * n * 4;

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
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes =
        |n: usize| n * ((b.h + b.hv) * t + 32) + 2 * n * 4 + KEYS * (b.h * t + 16) + KEYS * n * t;
    [32, 16].into_iter().find(|&n| bytes(n) <= 28 << 10)
}

/// Simdgroups of the forward kernel's threadgroup, each owning 8 queries: 8 (64 queries a block) when that
/// still makes 512 threadgroups, else 4 (32, as MLX's steel attention) to keep the GPU busy. On an M4 Pro, 8
/// heads of 64: S = 1024 runs 0.8x torch with 64-query blocks and 1.0-1.1x with 32; S = 4096 runs 1.07x with
/// 64 and 0.92x with 32, its keys and values read twice as often.
pub(crate) fn forward_groups(a: &Attention) -> usize {
    if mpp(a) {
        return 4;
    }

    let (outer, heads) = grid_batch(&a.batch);

    match a.sq.div_ceil(64) * heads * outer >= 512 {
        true => 8,
        false => 4,
    }
}

/// Query rows each forward simdgroup computes: 16 with Metal Performance Primitives, else 8.
pub(crate) fn forward_rows(a: &Attention) -> usize {
    match mpp(a) {
        true => 16,
        false => 8,
    }
}

/// Whether the forward runs on Metal Performance Primitives (`flash_attention_mpp`): half or bfloat16
/// operands with contiguous columns, head dims of whole 16-wide fragments, whole 16-query and 32-key blocks,
/// at least 4096 keys, no dropout, and a toolchain that has them. Below 4096 keys its round trip of scores
/// through threadgroup memory costs more than its matmuls save: on an M4 Pro, causal with 8 heads of 64 in
/// bfloat16, S = 4096 and 8192 run 5% faster than the 8x8 kernel, S = 256 to 2048 4-25% slower.
pub(crate) fn mpp(a: &Attention) -> bool {
    matches!(a.dtype, DType::F16 | DType::BF16)
        && !decodes(a)
        && a.dropout.is_none()
        && [&a.q, &a.k, &a.v].iter().all(|x| x.col == 1)
        && a.h.is_multiple_of(16)
        && a.hv.is_multiple_of(16)
        && a.sq.is_multiple_of(16)
        && a.sk.is_multiple_of(32)
        && a.sk >= 4096
        && crate::graph_compiler::mps::mpp_available()
}

/// Whether the forward uses the decode kernel (few query rows).
pub(crate) fn decodes(a: &Attention) -> bool {
    a.sq <= 8
}

/// Keys per block for the forward kernel, the largest whose K/V tiles fit in threadgroup memory.
pub(crate) fn key_block(a: &Attention) -> usize {
    let t = a.dtype.size_of();
    // Each tile row is padded by 16 bytes (`ATTN_PAD`).
    let bytes = |bk: usize| bk * ((a.h + a.hv) * t + 32);

    [32, 16, 8]
        .into_iter()
        .find(|&bk| bytes(bk) <= 28 << 10)
        .unwrap_or(8)
}
