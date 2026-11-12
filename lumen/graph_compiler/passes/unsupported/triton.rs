//! What Triton runs for this pass.

use super::Unsupported;
use crate::DType;
use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::pipeline::Pass;

/// Rejects graphs with half-precision transcendentals, which Triton's libdevice lacks.
fn unsupported(graph: &Graph) -> Result<(), String> {
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

/// Always: rejects graphs with an op the device has no kernel for.
impl DevicePass<Triton> for Unsupported {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Triton>> {
        Box::new(Unsupported { check: unsupported })
    }
}
