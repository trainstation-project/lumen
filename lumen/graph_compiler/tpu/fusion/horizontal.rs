//! Horizontal fusion of independent same-shape `elementwise` fusions into one kernel launch.
use super::super::TpuModule;
use crate::codegen::utils::tpu::fusible;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::horizontal::{Target, fuse};
use crate::graph_compiler::pass::CompilerPass;

/// Largest output, in elements, of an elementwise fusion eligible for horizontal fusion.
const MAX_ELEMENTS: usize = 1 << 20;

/// Most buffers one horizontally fused kernel may read and write.
const MAX_STREAMS: usize = 32;

/// Horizontal fusion target that merges small TPU `elementwise` fusions sharing an output shape.
struct Elementwise;

impl Target for Elementwise {
    /// Accepts non-scalar, small, fully fusible `elementwise` fusions not written in place into a concat.
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let ty = graph.type_of(node.output);

        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            return false;
        };

        let placed = graph.nodes().iter().any(|n| {
            matches!(n.primitive, Primitive::Concatenate { in_place: true, .. })
                && n.inputs.contains(&node.output)
        });

        name == "elementwise"
            && !ty.shape.is_empty()
            && ty.numel() <= MAX_ELEMENTS
            && body.nodes().iter().all(|n| fusible(&n.primitive))
            && !placed
    }

    /// Groups candidates by output shape.
    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize> {
        graph.type_of(node.output).shape.clone()
    }

    /// Caps the buffer count at `MAX_STREAMS`.
    fn fits(&self, _: &Graph, _: &[usize], buffers: usize) -> bool {
        buffers <= MAX_STREAMS
    }

    /// Accepts any group that fits.
    fn accepts(&self, _: &Graph, _: &[usize], _: &Graph) -> bool {
        true
    }

    /// Names the merged kernel `elementwise`.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        "elementwise".into()
    }
}

/// Merges independent small `elementwise` fusions with the same output shape.
pub(in crate::graph_compiler::tpu) struct HorizontalFusion;

impl CompilerPass<TpuModule<'_>> for HorizontalFusion {
    fn name(&self) -> &'static str {
        "horizontal_fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = fuse(&m.graph, &[], &mut Elementwise);
        Ok(())
    }
}
