//! TPU fusion: its planned-group fusion (`passes::fusion::groups`) into `elementwise` and `reduction` fusions, which
//! nodes horizontal fusion merges, and which producers write a concatenation's slots (its dot fusions live in
//! `passes::dot`).

use crate::codegen::utils::tpu::LANES;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::{DeviceModule, Tpu};

/// TPU's fusion (the graph compiler's `fusion` pass): its planned groups (`passes::fusion::groups`), always.
impl crate::codegen::Fuses for Tpu {
    const WITH_FUSE: bool = false;

    /// Rewrites the graph with each planned group fused into an `elementwise` or `reduction` fusion.
    fn fuse(m: &mut DeviceModule<'_, Tpu>) -> Result<(), String> {
        m.graph = crate::graph_compiler::passes::fusion::groups::elementwise_fusion(&m.graph);
        Ok(())
    }
}

/// Most buffers one horizontally fused TPU kernel may read and write.
pub(crate) const HORIZONTAL_BUFFERS: usize = 32;

/// Whether horizontal fusion may merge `node`: a non-scalar `elementwise` fusion of fully fusible nodes not
/// written in place into a concat.
pub(crate) fn horizontal_candidate(graph: &Graph, node: &Node) -> bool {
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

/// Whether operand `k` of `concat`, made by `producer` (a fusion when it is one of its outputs), can be written
/// straight into its slot: the concat joins its last axis, the operand's width fills whole lanes, and its
/// producer's kernel writes a strided slice of the shared buffer (a dot, an elementwise fusion, a fill, a
/// scalar broadcast or an elementwise op).
pub(crate) fn slot_writes(graph: &Graph, concat: &Node, k: usize, producer: &Node) -> bool {
    let Primitive::Concatenate { dimension, .. } = concat.primitive else {
        return false;
    };

    let writes = match &producer.primitive {
        Primitive::DotGeneral { .. } | Primitive::Full { .. } => true,
        Primitive::Fusion { name, .. } => name == "dot_general" || name == "elementwise",
        Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } => broadcast_dimensions.is_empty(),
        p => crate::graph_compiler::elementwise(p),
    };

    dimension + 1 == graph.type_of(concat.output).shape.len()
        && graph.type_of(concat.inputs[k]).shape[dimension].is_multiple_of(LANES)
        && writes
}
