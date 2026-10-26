//! Matmul kernel configuration shared by every Metal GEMM algorithm.

/// Which Metal matmul kernel a config runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Algorithm {
    /// Simdgroup-matrix tiles staged through threadgroup memory (`matmul_sg`).
    Tiled,
    /// Metal Performance Primitives tensor ops (`matmul_mpp`).
    Mpp,
    /// Few output rows: lanes split K and reduce partial sums (`matmul_skinny`).
    Skinny,
    /// Matrix-vector product with a simd reduction per output (`matmul_gemv`).
    Gemv,
}

/// Tile shape, thread count and algorithm of one matmul kernel variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmConfig {
    /// Output tile rows per threadgroup.
    pub bm: usize,
    /// Output tile columns per threadgroup.
    pub bn: usize,
    /// K elements consumed per step (for GEMV, elements each lane loads per step).
    pub bk: usize,
    /// Threads per threadgroup.
    pub threads: usize,
    /// Simdgroup columns the tile is split into (tiled algorithm).
    pub cols: usize,
    /// Whether the tiled kernel prefetches the next K tile while computing.
    pub prefetch: bool,
    pub algorithm: Algorithm,
}

/// Threadgroup memory available to one matmul threadgroup, in bytes.
pub(crate) const MAX_THREADGROUP_BYTES: usize = 32 << 10;
