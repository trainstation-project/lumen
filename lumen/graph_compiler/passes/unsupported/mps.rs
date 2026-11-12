//! What MPS runs for this pass.

use super::Unsupported;
use crate::DType;
use crate::graph::{Graph, Primitive, TensorType};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{DeviceModule, Mps};
use crate::graph_compiler::passes::DevicePass;
use crate::graph_compiler::pipeline::Pass;

/// Rejects graphs with ops MPS has no kernel for: bfloat16 transcendentals and mixed-dtype dots.
fn unsupported(graph: &Graph) -> Result<(), String> {
    use Primitive::*;

    // Metal has no float64 or float8 type for a kernel to compute or copy them in.
    let metal = |ty: &TensorType| {
        !matches!(
            ty.dtype,
            DType::F64 | DType::F8E4M3FN | DType::F8E5M2 | DType::F8E4M3B11FNUZ
        )
    };

    for node in graph.nodes() {
        let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));

        let typed = node
            .inputs
            .iter()
            .chain([&node.output])
            .map(|&v| graph.type_of(v));

        if let Some(ty) = typed.into_iter().find(|ty| !metal(ty)) {
            return Err(format!(
                "{}: {} is not supported on MPS",
                node.primitive.name(),
                ty.dtype.full_name()
            ));
        }

        if node.primitive.is_transcendental() && x.dtype == DType::BF16 {
            return Err(format!(
                "{}: has no bfloat16 kernel on MPS, got {x}: convert to float32 first (x.to(dtype=float32))",
                node.primitive.name()
            ));
        }

        if let DotGeneral { .. } = node.primitive {
            let y = graph.type_of(node.inputs[1]);

            if x.dtype != y.dtype {
                return Err(format!(
                    "dot_general: has no kernel of two dtypes on MPS, got {x} and {y}: convert them to one first"
                ));
            }
        }
    }

    Ok(())
}

/// Always: rejects graphs with an op the device has no kernel for.
impl DevicePass<Mps> for Unsupported {
    fn pass<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Mps>> {
        Box::new(Unsupported { check: unsupported })
    }
}
