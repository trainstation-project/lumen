//! TPU's answers to the `place_concats` and `concats` passes: which producers write a concatenation's slots.

use crate::codegen::devices::tpu::LANES;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::{DeviceModule, Tpu};

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

/// TPU's kernels writing a concatenation's operands into their slots, always: what `slot_writes` accepts, every operand
/// worth placing.
impl super::Slots for DeviceModule<'_, Tpu> {
    /// What `slot_writes` accepts.
    fn writes(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        _: usize,
    ) -> bool {
        crate::graph_compiler::passes::place_concats::tpu::slot_writes(
            &self.graph,
            concat,
            k,
            producer,
        )
    }

    /// Any operand: on TPU's tiled layout a concatenation fusion would build is a copy of its own.
    fn worth(&self, _: &crate::graph::Node, _: usize, _: &crate::graph::Node, _: usize) -> bool {
        true
    }
}

/// TPU rechecks its in-place concatenations after fusion: an operand fusion made a reduction's, or an elementwise
/// fusion's output other than its first (the only one its kernel writes into a window), is copied instead.
impl super::Rechecks for DeviceModule<'_, Tpu> {
    /// What `slot_writes` accepts, an elementwise fusion's first output only.
    fn written(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool {
        let first = match &producer.primitive {
            Primitive::Fusion { name, .. } => name != "elementwise" || index == 0,
            _ => true,
        };

        first && slot_writes(&self.graph, concat, k, producer)
    }
}
