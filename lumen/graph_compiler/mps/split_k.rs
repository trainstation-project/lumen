//! MPS split-K: the shared pass (`common::split_k`) with MPS's tiles, its GEMV and skinny (few-row)
//! products kept whole.

use crate::codegen::dot_general::mps::{float_groups, gemv, skinny_rows};
use crate::graph::Graph;
use crate::graph_compiler::common::split_k::Rules;
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Most rows of a skinny product kept whole: its lanes already split K, while split, each chunk runs on a
/// tiled GEMM of mostly-padding 32-row tiles. On an M4 Pro, bf16 [4, 4096] · [4096, 1024] runs 59 µs split
/// and 27.5 µs whole; at 16 and 32 rows splitting wins (bf16 [16, 16384] · [16384, 1024]: 178 vs 302 µs).
const WHOLE_ROWS: usize = 4;

/// The chunks MLX's split-K picks (`steel_gemm_splitk_axpby`): an output of at most 2048 32 x 32 tiles
/// with `k` at least 128 and at least `m` and `n` splits into `next_pow2((k / 16) / tiles)` chunks, 2 to
/// 32, more as K per tile grows. The pass takes it where it beats the 256-threadgroup target: on an M4
/// Pro, small outputs of long K run 6-38% faster (bf16 [64, 16384] · [16384, 64]: 32 chunks 29 µs, 8
/// chunks 40 µs), larger ones keep their count.
fn mlx_chunks(m: usize, n: usize, k: usize) -> Option<usize> {
    let tiles = m.div_ceil(32) * n.div_ceil(32);

    if tiles > 2048 || k / 16 < 8 || k < m.max(n) {
        return None;
    }

    Some((k / 16 / tiles).next_power_of_two().clamp(2, 32))
}

/// Splits `graph`'s matmuls by MPS's rules.
pub(super) fn split_k(graph: &Graph) -> Graph {
    let rules = Rules {
        tiles: &|m, n| float_groups(1, m, n),
        target: 256,
        unsplit: &|dot, lhs, rhs, out| {
            gemv(dot, lhs, rhs, out).is_some()
                || skinny_rows(dot, lhs, rhs, out).is_some_and(|m| m <= WHOLE_ROWS)
        },
        chunks: Some(&mlx_chunks),
    };

    crate::graph_compiler::common::split_k::split_k(graph, &rules)
}

/// Compiler pass running `split_k`.
pub(super) struct SplitK;

impl<M: Module> CompilerPass<M> for SplitK {
    fn name(&self) -> &'static str {
        "split_k"
    }

    /// Splits the module's matmuls.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = split_k(m.graph());
        Ok(())
    }
}
