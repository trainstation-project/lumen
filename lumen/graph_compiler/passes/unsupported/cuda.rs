//! CUDA's answer to the `unsupported` pass: which ops it has no kernel for.

use crate::codegen::devices::cuda::Cuda;
use crate::codegen::fusion::cuda::fusible;
use crate::graph::{Graph, Primitive};

/// What the CUDA (CuTe DSL) backend runs: elementwise and layout ops and gathers in CuTe dtypes (`fusible`),
/// matmuls, `take_along_axis` (lowered to a gather), and the reductions of an attention `flash_attention` fuses
/// (any other reduction still fails at codegen). Anything else runs on Triton.
impl super::Supports for Cuda {
    /// Rejects graphs with an op the CuTe DSL backend has no kernel for, or a dtype CuTe has no type for.
    fn check(graph: &Graph) -> Result<(), String> {
        for node in graph.nodes() {
            let runs = fusible(graph, node)
                || matches!(
                    node.primitive,
                    Primitive::DotGeneral { .. }
                        | Primitive::TakeAlongAxis { .. }
                        | Primitive::ReduceSum { .. }
                        | Primitive::ReduceMax { .. }
                );

            if !runs {
                return Err(format!(
                    "{}: the CUDA (CuTe DSL) backend has no kernel for it yet; turn \
                     config.compiler.use_triton_backend on to run it on Triton",
                    node.primitive.name()
                ));
            }
        }

        Ok(())
    }
}
