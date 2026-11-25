//! TPU's answers to the `horizontal_fusion` pass: which kernels it merges, grouped how, binding how many buffers.

use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::Tpu;

/// Most buffers one horizontally fused TPU kernel may read and write.
const HORIZONTAL_BUFFERS: usize = 32;

/// Whether horizontal fusion may merge `node`: a non-scalar `elementwise` fusion of fully fusible nodes not
/// written in place into a concat.
fn horizontal_candidate(graph: &Graph, node: &Node) -> bool {
    let Primitive::Fusion { name, body, .. } = &node.primitive else {
        return false;
    };

    let placed = graph.nodes().iter().any(|n| {
        matches!(n.primitive, Primitive::Concatenate { in_place: true, .. })
            && n.inputs.contains(&node.output)
    });

    name == "elementwise"
        && !graph.type_of(node.output).shape.is_empty()
        && body
            .nodes()
            .iter()
            .all(|n| crate::codegen::devices::tpu::fusible(&n.primitive))
        && !placed
}

/// TPU merges non-scalar `elementwise` fusions grouped by output shape, without `fuse` too.
impl super::HorizontallyFuses for Tpu {
    const GROUPING: super::Grouping = super::Grouping::Shape;
    const BUFFERS: usize = HORIZONTAL_BUFFERS;

    /// `horizontal_candidate`.
    fn candidate(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool {
        horizontal_candidate(graph, node)
    }
}
