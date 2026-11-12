//! TPU fusion passes: elementwise/reduction fusion and horizontal fusion (its dot fusions live in
//! `passes::dot`).

use crate::codegen::utils::tpu::LANES;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Tpu};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::fusion::FusionDevice;
use crate::graph_compiler::pipeline::Pass;
mod emit;
mod groups;

/// Groups elementwise and reduction nodes into `elementwise` and `reduction` fusions.
pub(crate) struct Fusion;

impl CompilerPass<DeviceModule<'_, Tpu>> for Fusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn apply(&mut self, m: &mut DeviceModule<'_, Tpu>) -> Result<(), String> {
        m.graph = emit::elementwise_fusion(&m.graph);
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

/// TPU's answers to the pass.
impl FusionDevice for Tpu {
    /// Always.
    fn fusion<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Fusion)
    }
}
