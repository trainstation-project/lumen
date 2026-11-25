//! MPS's answers to the `place_concats` and `concats` passes: which producers write a concatenation's slots.

use crate::codegen::devices::mps::Kernel;
use crate::codegen::fusion::mps::writes_views;
use crate::graph::{Node, Primitive};
use crate::graph_compiler::device::{DeviceModule, Mps};

/// The kind of kernel `producer` awaits when it is a still-pending fusion.
fn pending(producer: &Node) -> Option<Kernel> {
    match &producer.primitive {
        Primitive::Fusion { name, .. } => Kernel::of(name),
        _ => None,
    }
}

/// Whether the kernel making operand `k` of `concat` (`producer`, its output `index`) can write it into its slot:
/// the attention backward kernel (its atomically accumulated dQ aside), a fusion writing views, or, before fusion
/// (`before_fusion`), a fusible layout or elementwise node.
fn slot_writes(
    m: &DeviceModule<'_, Mps>,
    concat: &Node,
    k: usize,
    producer: &Node,
    index: usize,
    before_fusion: bool,
) -> bool {
    match (pending(producer), &producer.primitive) {
        (Some(Kernel::AttentionBackward), Primitive::Fusion { body, .. }) => {
            let atomic = body.outputs().len() == 3 && index == 2;
            let at = m.graph.slot(concat, k);
            let views = std::collections::HashMap::from([(index, (at.offset, at.strides))]);
            !atomic
                && crate::graph_compiler::passes::attention::backward_of_body_into(body, views)
                    .is_some()
        }
        (Some(Kernel::Fusion), Primitive::Fusion { body, .. }) => writes_views(body),
        (None, p) if before_fusion => {
            let layout = matches!(
                p,
                Primitive::Reshape { .. }
                    | Primitive::Transpose { .. }
                    | Primitive::BroadcastInDim { .. }
                    | Primitive::Slice { .. }
            );

            crate::codegen::fusion::mps::fusible(&m.graph, producer)
                && (layout || crate::graph_compiler::elementwise(p))
        }
        _ => false,
    }
}

/// Metal's kernels writing a concatenation's operands into their slots, before fusion (`slot_writes`, fusible nodes
/// counting as kernels), with `fuse`.
impl super::Slots for DeviceModule<'_, Mps> {
    /// What `slot_writes` accepts before fusion.
    fn writes(&self, concat: &Node, k: usize, producer: &Node, index: usize) -> bool {
        slot_writes(self, concat, k, producer, index, true)
    }
}

/// MPS rechecks its in-place concatenations after fusion: some fusion leaves unable to write their slot.
impl super::Rechecks for DeviceModule<'_, Mps> {
    /// What `slot_writes` accepts after fusion.
    fn written(&self, concat: &Node, k: usize, producer: &Node, index: usize) -> bool {
        slot_writes(self, concat, k, producer, index, false)
    }
}
