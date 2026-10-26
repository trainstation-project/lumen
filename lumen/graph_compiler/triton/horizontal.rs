//! Horizontal fusion of independent small elementwise `loop` fusions into one Triton launch.

use super::TritonModule;
use crate::codegen::triton as codegen;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::common::horizontal::{Target, fuse};
use crate::graph_compiler::pass::CompilerPass;

/// Largest output, in elements, of a loop fusion eligible for horizontal fusion.
const MAX_ELEMENTS: usize = 1 << 20;

/// Most tensors one horizontally fused kernel may read and write.
const MAX_BUFFERS: usize = 64;

/// Horizontal fusion target that merges small elementwise Triton `loop` fusions of one element count.
struct Loops;

impl Target for Loops {
    /// Accepts non-empty, small `loop` fusions without a reduction or GEMM.
    fn candidate(&self, graph: &Graph, node: &Node) -> bool {
        let elements = graph.type_of(node.output).numel();
        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            return false;
        };
        name == "loop"
            && elements > 0
            && elements <= MAX_ELEMENTS
            && codegen::reduction_root(body).is_none()
            && codegen::gemm_dot(body).is_none()
    }

    /// Groups candidates by element count, the one flat grid every output shares.
    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize> {
        vec![graph.type_of(node.output).numel()]
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

/// Merges independent small elementwise `loop` fusions with the same element count.
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
