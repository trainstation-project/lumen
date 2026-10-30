//! MPS target for horizontal fusion: merges independent small elementwise kernels into one launch.

use super::{codegen, fusion, split_k};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::common::horizontal::{Target, heroes_fit, keeps_tiling, tiling_key};

/// Most buffers one horizontally fused kernel may bind.
const MAX_BUFFERS: usize = 30;

/// Largest output, in elements, of a node worth fusing horizontally.
const MAX_ELEMENTS: usize = 1 << 20;

/// Horizontal fusion target that names merged kernels through `kernel`.
pub(super) struct Horizontal<'a> {
    /// Generates the kernel for a merged body and its by-value flags, returning its name.
    pub(super) kernel: &'a mut dyn FnMut(&Graph, &[bool]) -> String,
}

impl Target for Horizontal<'_> {
    /// Small elementwise nodes or loop fusions without reductions, attention, GEMMs or updates.
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let elements = graph.type_of(node.output).numel();

        if elements == 0 || elements > MAX_ELEMENTS {
            return false;
        }

        match &node.primitive {
            Primitive::Fusion { name, .. } if name == crate::graph::UPDATES => false,
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

    /// Groups loops of any size together (they split the grid), and tiled kernels by exact shape.
    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize> {
        tiling_key(graph, node)
    }

    /// Whether a group stays under the buffer limit and, for tiled kernels, the transpose limit.
    fn fits(&self, graph: &Graph, members: &[usize], buffers: usize) -> bool {
        heroes_fit(graph, members) && buffers <= MAX_BUFFERS
    }

    /// Whether the merged body keeps the members' tiling.
    fn accepts(&self, graph: &Graph, members: &[usize], body: &Graph) -> bool {
        keeps_tiling(graph, members, body)
    }

    /// Generates the merged kernel.
    fn kernel(&mut self, body: &Graph, by_value: &[bool]) -> String {
        (self.kernel)(body, by_value)
    }
}
