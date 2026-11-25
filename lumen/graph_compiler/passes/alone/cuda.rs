//! CUDA's answers to the `alone` pass: which lone nodes only a generated kernel computes, and the kernels' names.

use crate::codegen::devices::cuda::Cuda;
use crate::codegen::fusion::cuda::{ELEMENTWISE, fusible};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::DeviceModule;

impl super::Lone for DeviceModule<'_, Cuda> {
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
