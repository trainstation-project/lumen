//! Triton's answers to the `horizontal_fusion` pass: which kernels it merges, grouped how, binding how many buffers.

use crate::codegen::kinds::{gemm_dot, reduction_root};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::Triton;

/// Most tensors one horizontally fused Triton kernel may read and write.
const HORIZONTAL_BUFFERS: usize = 64;

/// Whether horizontal fusion may merge `node`: an `elementwise` fusion without reductions or matmuls, as
/// MPS's are; a node alone joins only when elementwise (MPS runs a lone fill, iota or the like as its own
/// step).
fn horizontal_candidate(_: &Graph, node: &Node) -> bool {
    let Primitive::Fusion { name, body, .. } = &node.primitive else {
        return false;
    };

    let alone = match body.nodes() {
        [n] => !crate::graph_compiler::elementwise(&n.primitive),
        _ => false,
    };

    name == "elementwise" && !alone && reduction_root(body).is_none() && gemm_dot(body).is_none()
}

/// Triton merges, with `fuse`, non-trivial `elementwise` fusions without reductions or matmuls, grouped by tiling.
impl super::HorizontallyFuses for Triton {
    const GROUPING: super::Grouping = super::Grouping::Tiling;
    const BUFFERS: usize = HORIZONTAL_BUFFERS;

    /// `horizontal_candidate`.
    fn candidate(graph: &Graph, node: &Node) -> bool {
        horizontal_candidate(graph, node)
    }
}
