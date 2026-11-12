//! What CUDA (CuTe DSL) runs for this pass.

use super::AloneDevice;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::alone::{Alone, Lone};
use crate::graph_compiler::passes::fusion::cuda::{ELEMENTWISE, fusible};
use crate::graph_compiler::pipeline::Pass;

/// CUDA (CuTe DSL)'s answers to the pass.
impl AloneDevice for Cuda {
    /// Always.
    fn alone<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Alone)
    }
}

impl Lone for DeviceModule<'_, Cuda> {
    /// Whether lone `n` needs a one-node kernel: every fusible node but reshapes, which are views.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n) && !matches!(n.primitive, Primitive::Reshape { .. })
    }

    /// The CUDA backend passes no inputs by value: runtime scalars are 0-d device tensors.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Every kernel is an elementwise one.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        ELEMENTWISE.into()
    }
}
