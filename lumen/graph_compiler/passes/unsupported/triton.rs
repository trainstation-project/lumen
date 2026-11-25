//! Triton's answer to the `unsupported` pass: which ops it has no kernel for.

use crate::DType;
use crate::codegen::devices::triton::Triton;
use crate::graph::Graph;

impl super::Supports for Triton {
    /// Rejects graphs with half-precision transcendentals, which Triton's libdevice lacks.
    fn check(graph: &Graph) -> Result<(), String> {
        for node in graph.nodes() {
            let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));

            if node.primitive.is_transcendental() && matches!(x.dtype, DType::F16 | DType::BF16) {
                return Err(format!(
                    "{}: has no {} kernel on Triton, got {x}: convert to float32 first (x.to(dtype=float32))",
                    node.primitive.name(),
                    x.dtype.name()
                ));
            }
        }

        Ok(())
    }
}
