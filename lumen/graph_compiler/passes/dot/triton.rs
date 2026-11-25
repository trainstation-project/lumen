//! Triton's answers to the dot passes.

use crate::codegen::devices::triton::Triton;
use crate::graph::{Graph, Primitive};

/// Tiles a Triton split aims for: on an H100, outputs of 64 or more 128x128 tiles already keep the GPU busy (a
/// 1024x1024 bfloat16 matmul is slower split), while 16 tiles gain about 2x.
const SPLIT_TARGET: usize = 128;

/// Triton splits with the untuned GEMM config's tiles, aiming at `SPLIT_TARGET`, keeping matrix-vector products whole
/// for the GEMV kernel (its lanes already split K, as MPS keeps its GEMVs). Its partials are always summed by a
/// reduction (no atomics), so results stay deterministic.
impl super::split_k::SplitsK for Triton {
    const TARGET: usize = SPLIT_TARGET;

    /// Tiles of the untuned GEMM config.
    fn tiles(m: usize, n: usize) -> usize {
        use crate::codegen::dot_general::triton::{CONFIGS, UNTUNED};
        let tile = CONFIGS[UNTUNED];
        m.div_ceil(tile.block_m) * n.div_ceil(tile.block_n)
    }

    /// Matrix-vector products (`dot_general::triton::gemv`).
    fn unsplit(
        dot: &Primitive,
        lhs: &crate::graph::TensorType,
        rhs: &crate::graph::TensorType,
        _: &crate::graph::TensorType,
    ) -> bool {
        crate::codegen::dot_general::triton::gemv(dot, lhs, rhs)
    }
}

/// Triton's GEMM writes a wide row's partials over `partial_tile`-column blocks, except on the GEMV kernel.
impl super::wide_rows::PartialRows for Triton {
    /// `dot_general::triton::partial_tile`.
    fn partial_tile(n: usize) -> Option<usize> {
        crate::codegen::dot_general::triton::partial_tile(n)
    }

    /// `dot_general::triton::gemv_body`.
    fn gemv(body: &Graph) -> bool {
        crate::codegen::dot_general::triton::gemv_body(body)
    }
}
