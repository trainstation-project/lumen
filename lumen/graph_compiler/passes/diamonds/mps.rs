//! MPS's answers to the `normalization_diamonds` pass: the nodes its row kernels take.

use crate::codegen::devices::mps::*;
use crate::graph::{Graph, Node};

/// MPS fusion fuses rows of what it fuses, column sums riding along within `row::mps::ROW_PARTIALS`.
impl super::NormalizationRows for Mps {
    const PARTIALS: super::Partials = crate::codegen::row::mps::ROW_PARTIALS;

    /// `fusion::mps::fusible`.
    fn fusible(graph: &Graph, node: &Node) -> bool {
        crate::codegen::fusion::mps::fusible(graph, node)
    }
}
