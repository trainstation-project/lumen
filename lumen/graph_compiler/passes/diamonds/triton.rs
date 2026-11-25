//! Triton's answers to the `normalization_diamonds` pass: the nodes its row kernels take.

use crate::codegen::devices::triton::*;
use crate::graph::{Graph, Node};

/// Triton's fusion fuses rows of what it fuses (the rows MPS finds), column sums riding along within
/// `row::triton::ROW_PARTIALS`.
impl super::NormalizationRows for Triton {
    const PARTIALS: super::Partials = crate::codegen::row::triton::ROW_PARTIALS;

    /// `fusion::triton::fusible`.
    fn fusible(graph: &Graph, node: &Node) -> bool {
        crate::codegen::fusion::triton::fusible(graph, node)
    }
}
