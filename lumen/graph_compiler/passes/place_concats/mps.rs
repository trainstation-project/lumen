//! In-place concatenation on MPS: which kernels write an operand straight into its slot, and the pass, after
//! fusion, copying the concatenations fusion left unable to be written in place.

use std::collections::HashMap;

use super::in_place_concats;
use crate::codegen::mps::writes_views;
use crate::graph::{Node, Primitive};
use crate::graph_compiler::mps::{Kernel, MpsModule, fusion, request};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::alone::wrapped;
use crate::graph_compiler::passes::attention;

/// Compiler pass copying the concatenations fusion left unable to be written in place.
pub(crate) struct Concats;

impl<'a> CompilerPass<MpsModule<'a>> for Concats {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "concats"
    }

    /// Unmarks in-place concatenations that fusion invalidated; each gets its own elementwise kernel copying it.
    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let module = &*m;

        let valid = in_place_concats(
            &m.graph,
            |c, k, p, index| slot_writes(module, c, k, p, index, false),
            |_, _, p, _| slot_worth(p),
        );
        let mut unmarked = false;

        for (i, node) in m.graph.nodes.iter_mut().enumerate() {
            if let Primitive::Concatenate { in_place, .. } = &mut node.primitive
                && *in_place
                && !valid.contains(&i)
            {
                *in_place = false;
                unmarked = true;
            }
        }

        if unmarked {
            let concatenate = |n: &crate::graph::Node| {
                matches!(
                    n.primitive,
                    Primitive::Concatenate {
                        in_place: false,
                        ..
                    }
                )
            };

            m.graph = wrapped(&m.graph, &[], concatenate, |_, _| request(Kernel::Fusion));
        }

        Ok(())
    }
}

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
pub(crate) fn slot_writes(
    m: &MpsModule<'_>,
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
            let views = HashMap::from([(index, (at.offset, at.strides))]);
            !atomic && attention::backward_of_body_into(body, views).is_some()
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

            fusion::fusible(&m.graph, producer) && (layout || fusion::elementwise(p))
        }
        _ => false,
    }
}

/// Whether an operand `producer` writes into its slot is worth placing its concatenation for: the attention
/// backward kernel's are (its gradients land side by side), others alone aren't.
pub(crate) fn slot_worth(producer: &Node) -> bool {
    matches!(pending(producer), Some(Kernel::AttentionBackward))
}
