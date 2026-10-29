//! Horizontal fusion of independent small elementwise `loop` fusions into one Triton launch;
//! members of one element count share a grid range, others get their own.

use super::TritonModule;
use crate::codegen::triton as codegen;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::horizontal::{Target, fuse};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::CompilerPass;

/// Largest output, in elements, of a loop fusion eligible for horizontal fusion.
const MAX_ELEMENTS: usize = 1 << 20;

/// Most tensors one horizontally fused kernel may read and write.
const MAX_BUFFERS: usize = 64;

/// Horizontal fusion target that merges small elementwise Triton `loop` fusions.
struct Loops;

impl Target for Loops {
    /// Accepts non-empty, small `loop` fusions without a reduction or GEMM, as MPS's target does: a node
    /// alone joins only when elementwise (MPS runs a lone fill, iota or the like as its own step).
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let elements = graph.type_of(node.output).numel();
        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            return false;
        };
        let alone = match body.nodes() {
            [n] => !elementwise(&n.primitive),
            _ => false,
        };
        name == "loop"
            && elements > 0
            && elements <= MAX_ELEMENTS
            && !alone
            && codegen::reduction_root(body).is_none()
            && codegen::gemm_dot(body).is_none()
    }

    /// Groups every candidate together: members of different element counts split the grid.
    fn key(&self, _: &Graph, _: &Node) -> Vec<usize> {
        Vec::new()
    }

    /// Caps the tensor count at `MAX_BUFFERS`.
    fn fits(&self, _: &Graph, _: &[usize], buffers: usize) -> bool {
        buffers <= MAX_BUFFERS
    }

    /// Accepts any group that fits.
    fn accepts(&self, _: &Graph, _: &[usize], _: &Graph) -> bool {
        true
    }

    /// Names the merged kernel `loop`; its source is generated per plan step.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        "loop".into()
    }
}

/// Merges independent small elementwise `loop` fusions.
pub(super) struct HorizontalFusion;

impl CompilerPass<TritonModule<'_>> for HorizontalFusion {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "horizontal_fusion"
    }

    /// Replaces each group of candidates with one multi-output `loop` fusion.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        m.graph = fuse(&m.graph, &[], &mut Loops);
        Ok(())
    }
}
