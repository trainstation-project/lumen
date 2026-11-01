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

/// Splits `graph`'s matmuls by MPS's rules.
pub(super) fn split_k(graph: &Graph) -> Graph {
    let rules = Rules {
        tiles: &|m, n| float_groups(1, m, n),
        target: 256,
        unsplit: &|dot, lhs, rhs, out| {
            gemv(dot, lhs, rhs, out).is_some()
                || skinny_rows(dot, lhs, rhs, out).is_some_and(|m| m <= WHOLE_ROWS)
        },
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
