use super::{codegen, fusion, split_k};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::attention;
use crate::graph_compiler::horizontal::Target;

const MAX_BUFFERS: usize = 30;

const MAX_ELEMENTS: usize = 1 << 20;

pub(super) struct Horizontal<'a> {
    pub(super) kernel: &'a mut dyn FnMut(&Graph, &[bool]) -> String,
}

fn tiled(node: &Node) -> bool {
    match &node.primitive {
        Primitive::Fusion { body, .. } => codegen::transpose_tiling(body).is_some(),
        _ => false,
    }
}

impl Target for Horizontal<'_> {
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let elements = graph.type_of(node.output).numel();
        if elements == 0 || elements > MAX_ELEMENTS {
            return false;
        }
        match &node.primitive {
            Primitive::Fusion { body, .. } => {
                let reduces = body.nodes().iter().any(|n| {
                    matches!(
                        n.primitive,
                        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                    )
                });
                !reduces
                    && attention::of_body(body).is_none()
                    && attention::backward_of_body(body).is_none()
                    && split_k::atomic_dot(body).is_none()
                    && codegen::gemm_dot(body).is_none()
            }
            p => fusion::elementwise(p) && fusion::fusible(graph, node),
        }
    }

    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize> {
        let ty = graph.type_of(node.output);
        let mut key = vec![ty.numel(), usize::from(tiled(node))];
        if tiled(node) {
            key.extend(&ty.shape);
        }
        key
    }

    fn fits(&self, graph: &Graph, members: &[usize], buffers: usize) -> bool {
        let nodes = graph.nodes();
        let transposes = |m: &usize| match &nodes[*m].primitive {
            Primitive::Fusion { body, .. } => body
                .nodes()
                .iter()
                .filter(|n| matches!(n.primitive, Primitive::Transpose { .. }))
                .count(),
            _ => 0,
        };
        let heroes = !tiled(&nodes[members[0]])
            || members.iter().map(transposes).sum::<usize>() <= codegen::MAX_HEROES;
        heroes && buffers <= MAX_BUFFERS
    }

    fn accepts(&self, graph: &Graph, members: &[usize], body: &Graph) -> bool {
        codegen::transpose_tiling(body).is_some() == tiled(&graph.nodes()[members[0]])
    }

    fn kernel(&mut self, body: &Graph, by_value: &[bool]) -> String {
        (self.kernel)(body, by_value)
    }
}
