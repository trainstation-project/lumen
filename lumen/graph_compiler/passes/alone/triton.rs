//! What Triton runs for this pass.

use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::passes::alone::{Alone, Lone};
use crate::graph_compiler::passes::fusion::triton::{fusible, named};
use crate::graph_compiler::pipeline::Pass;

impl Lone for DeviceModule<'_, Triton> {
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
        named(body)
    }
}

/// Always.
impl DevicePass<Triton> for Alone {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        Box::new(Alone)
    }
}
