//! TPU's answers to the `normalization_diamonds` pass: the nodes its row kernels take.

use crate::codegen::devices::tpu::Tpu;

/// TPU fuses each row its row kernel takes into a `row` fusion of its own.
impl super::FusedRows for Tpu {
    /// `codegen::row::tpu::fusible`.
    fn fusible(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool {
        crate::codegen::row::tpu::fusible(graph, node)
    }

    /// `codegen::row::tpu::fits`.
    fn fits(body: &crate::graph::Graph) -> bool {
        crate::codegen::row::tpu::fits(body)
    }
}
