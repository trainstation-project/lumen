//! TPU fusion passes: dot epilogues, row-wise dot fusion, elementwise/reduction fusion and horizontal fusion.
mod dot_epilogues;
mod dot_rows;
mod emit;
mod groups;

pub(super) use dot_epilogues::FuseDots;
pub(super) use dot_rows::DotRows;

use super::TpuModule;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::pass::CompilerPass;

/// Groups elementwise and reduction nodes into `elementwise` and `reduction` fusions.
pub(super) struct Fusion;

impl CompilerPass<TpuModule<'_>> for Fusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = emit::elementwise_fusion(&m.graph);
        Ok(())
    }
}

/// Most buffers one horizontally fused TPU kernel may read and write.
pub(super) const HORIZONTAL_BUFFERS: usize = 32;

/// Whether horizontal fusion may merge `node`: a non-scalar `elementwise` fusion of fully fusible nodes not
/// written in place into a concat.
pub(super) fn horizontal_candidate(graph: &Graph, node: &Node) -> bool {
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
            .all(|n| crate::codegen::utils::tpu::fusible(&n.primitive))
        && !placed
}
