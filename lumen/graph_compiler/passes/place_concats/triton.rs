//! Triton's answers to the `place_concats` and `concats` passes: which producers write a concatenation's slots.

use crate::codegen::fusion::triton::fusible;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::{DeviceModule, Triton};

/// Whether `producer` ends a kernel of its own writing fresh memory, so it can write a concatenation's operand
/// straight into its slot through the slot's strides: a matmul, or a fusible elementwise or copying layout op
/// (a reshape or view slice aliases its operand instead). Fusion leaves such a concatenate unfused, so each
/// producer is its kernel's root.
fn slot_writes(graph: &Graph, producer: &Node) -> bool {
    let copies = matches!(
        producer.primitive,
        Primitive::Transpose { .. }
            | Primitive::BroadcastInDim { .. }
            | Primitive::Slice { view: false, .. }
    );

    let kernel = fusible(graph, producer)
        && (crate::graph_compiler::elementwise(&producer.primitive) || copies);

    kernel || matches!(producer.primitive, Primitive::DotGeneral { .. })
}

/// Triton's kernels writing a concatenation's operands into their slots, with `fuse`: kernels of their own writing
/// fresh memory (`slot_writes`).
impl super::Slots for DeviceModule<'_, Triton> {
    /// What `slot_writes` accepts.
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        slot_writes(&self.graph, producer)
    }
}
