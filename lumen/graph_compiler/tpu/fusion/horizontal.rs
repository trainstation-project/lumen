use super::super::TpuModule;
use crate::codegen::tpu::fusible;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::horizontal::{Target, fuse};
use crate::graph_compiler::pass::CompilerPass;

const MAX_ELEMENTS: usize = 1 << 20;

const MAX_STREAMS: usize = 32;

struct Loops;

impl Target for Loops {
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let ty = graph.type_of(node.output);
        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            return false;
        };
        let placed = graph.nodes().iter().any(|n| {
            matches!(n.primitive, Primitive::Concatenate { in_place: true, .. })
                && n.inputs.contains(&node.output)
        });
        name == "loop"
            && !ty.shape.is_empty()
            && ty.numel() <= MAX_ELEMENTS
            && body.nodes().iter().all(|n| fusible(&n.primitive))
            && !placed
    }

    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize> {
        graph.type_of(node.output).shape.clone()
    }

    fn fits(&self, _: &Graph, _: &[usize], buffers: usize) -> bool {
        buffers <= MAX_STREAMS
    }

    fn accepts(&self, _: &Graph, _: &[usize], _: &Graph) -> bool {
        true
    }

    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        "loop".into()
    }
}

pub(in crate::graph_compiler::tpu) struct HorizontalFusion;

impl CompilerPass<TpuModule<'_>> for HorizontalFusion {
    fn name(&self) -> &'static str {
        "horizontal_fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = fuse(&m.graph, &[], &mut Loops);
        Ok(())
    }
}
