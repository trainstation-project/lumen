//! MPS's answers to the dot passes.

use crate::codegen::devices::mps::Mps;
use crate::graph::Primitive;

/// Metal's GEMMs read operands laid out with each batch, free and contracting group merging into one stride.
impl super::canonicalize::DotLayouts for Mps {
    /// What `dot_general::mps::reads_strided` accepts at the operand's contiguous strides.
    fn readable(dot: &Primitive, operands: [&crate::graph::TensorType; 2], k: usize) -> bool {
        let strides = crate::tensor::contiguous_strides(&operands[k].shape);
        crate::codegen::dot_general::mps::reads_strided(dot, operands, k, &strides)
    }
}

/// Most rows of a skinny product kept whole: its lanes already split K, while split, each chunk runs on a
/// tiled GEMM of mostly-padding 32-row tiles. On an M4 Pro, bf16 [4, 4096] · [4096, 1024] runs 59 µs split
/// and 27.5 µs whole; at 16 and 32 rows splitting wins (bf16 [16, 16384] · [16384, 1024]: 178 vs 302 µs).
const WHOLE_ROWS: usize = 4;

/// The chunks MLX's split-K picks (`steel_gemm_splitk_axpby`): an output of at most 2048 32 x 32 tiles
/// with `k` at least 128 and at least `m` and `n` splits into `next_pow2((k / 16) / tiles)` chunks, 2 to
/// 32, more as K per tile grows. The pass takes it where it beats the 256-threadgroup target: on an M4
/// Pro, small outputs of long K run 6-38% faster (bf16 [64, 16384] · [16384, 64]: 32 chunks 29 µs, 8
/// chunks 40 µs), larger ones keep their count.
fn chunks(m: usize, n: usize, k: usize) -> Option<usize> {
    let tiles = m.div_ceil(32) * n.div_ceil(32);

    if tiles > 2048 || k / 16 < 8 || k < m.max(n) {
        return None;
    }

    Some((k / 16 / tiles).next_power_of_two().clamp(2, 32))
}

/// MPS splits with its GEMM's tiles (`float_groups`), aiming at 256 threadgroups, keeping GEMV and skinny products of
/// at most `WHOLE_ROWS` rows whole, and preferring MLX's chunk counts (`chunks`).
impl super::split_k::SplitsK for Mps {
    const TARGET: usize = 256;

    /// Threadgroups of the GEMM's float tiles.
    fn tiles(m: usize, n: usize) -> usize {
        crate::codegen::dot_general::mps::float_groups(1, m, n)
    }

    /// GEMV and skinny products of at most `WHOLE_ROWS` rows.
    fn unsplit(
        dot: &Primitive,
        lhs: &crate::graph::TensorType,
        rhs: &crate::graph::TensorType,
        out: &crate::graph::TensorType,
    ) -> bool {
        use crate::codegen::dot_general::mps::{gemv, skinny_rows};
        gemv(dot, lhs, rhs, out).is_some()
            || skinny_rows(dot, lhs, rhs, out).is_some_and(|m| m <= WHOLE_ROWS)
    }

    /// MLX's chunk count.
    fn chunks(m: usize, n: usize, k: usize) -> Option<usize> {
        chunks(m, n, k)
    }
}
