//! The CPU's answer to the `unsupported` pass: which ops it has no kernel for.

use crate::graph::{Graph, Primitive};
use crate::graph_compiler::device::Cpu;

/// What the CPU (this interpreter) runs: every primitive but a `lumen.kernels.pallas` kernel, which runs on a TPU.
impl super::Supports for Cpu {
    /// Rejects graphs with a Pallas kernel.
    fn check(graph: &Graph) -> Result<(), String> {
        match graph
            .nodes()
            .iter()
            .find(|n| matches!(n.primitive, Primitive::PallasCall { .. }))
        {
            Some(n) => Err(format!(
                "{}: a lumen.kernels.pallas kernel runs on a TPU, not the CPU",
                n.primitive.name()
            )),
            None => Ok(()),
        }
    }
}
