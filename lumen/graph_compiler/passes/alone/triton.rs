//! Triton's answers to the `alone` pass: which lone nodes only a generated kernel computes, and the kernels' names.

use crate::codegen::devices::triton::Triton;
use crate::codegen::fusion::triton::fusible;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::DeviceModule;

impl super::Lone for DeviceModule<'_, Triton> {
    /// Whether lone `n` is one a Triton loop kernel computes (Triton has no other kernel for it): every
    /// fusible node but views of their operand (reshapes, slices placed as views) and in-place
    /// concatenations.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n)
            && !matches!(
                n.primitive,
                Primitive::Reshape { .. }
                    | Primitive::Slice { view: true, .. }
                    | Primitive::Concatenate { in_place: true, .. }
            )
    }

    /// Triton passes no inputs by value.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Names the kernel by the body's kind.
    fn kernel(&mut self, body: &Graph, _: &[bool]) -> String {
        crate::codegen::fusion::triton::named(body)
    }
}
