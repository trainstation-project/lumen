//! What CUDA (CuTe DSL) runs for this pass.

use crate::codegen::devices::cuda::cute_type;
use crate::codegen::dot_general::cuda::epilogue;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::regions::expansion;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::device::{Cuda, DeviceModule};
use crate::graph_compiler::passes::fusion::{Backend, fuse};

/// Name of an elementwise fusion.
pub(crate) const ELEMENTWISE: &str = "elementwise";

/// Name of a GEMM fusion with its epilogue.
const DOT_GENERAL: &str = "dot_general";

/// Whether `node` can be computed inside a CUDA elementwise kernel: an elementwise or layout op
/// (or a fill, an iota, a concatenation, a pair swap or a gather) of dtypes CuTe has.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use crate::graph::{BinaryOp, UnaryOp};
    use Primitive::*;

    let op = match &node.primitive {
        Add
        | Sub
        | Mul
        | Div
        | Max
        | Eq
        | Lt
        | Neg
        | Exp
        | Log
        | Sqrt
        | Tanh
        | Logistic
        | Cast { .. }
        | Select
        | Reshape { .. }
        | BroadcastInDim { .. }
        | Transpose { .. }
        | Slice { .. }
        | Full { .. }
        | Iota { .. }
        | Concatenate { .. }
        | SwapPairs { .. }
        | Gather { .. } => true,
        Unary(op) => !matches!(
            op,
            UnaryOp::PopulationCount | UnaryOp::Clz | UnaryOp::ReducePrecision { .. }
        ),
        Binary(op) => !matches!(op, BinaryOp::Nextafter),
        _ => false,
    };

    let typed = |v: &Var| cute_type(graph.type_of(*v).dtype).is_some();
    op && node.inputs.iter().chain([&node.output]).all(typed)
}

/// The shared fusion's view of the CUDA backend: elementwise loops and GEMM epilogues (no reductions or rows yet),
/// any number of tensors, a fusion writing the values others read too, a GEMM epilogue at most 4 values besides its
/// last (`codegen::dot_general::cuda::epilogue` says which matmuls run one), its result an expansion as on Triton.
fn backend(m: &DeviceModule<'_, Cuda>) -> Backend {
    let config = &m.options.config;

    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: false,
        contraction_epilogues: config.contraction_epilogues,
        dot_epilogue: epilogue,
        expansion: |graph, dot, concat| expansion(graph, dot, concat).is_some(),
        row_dot: |_| false,
        swap_epilogues: false,
        combine_epilogues: false,
        update_epilogues: false,
    }
}

/// The name of a fused `body`: a GEMM with its epilogue, or an elementwise fusion.
fn named(body: &Graph) -> String {
    match gemm_dot(body) {
        Some(_) => DOT_GENERAL,
        None => ELEMENTWISE,
    }
    .into()
}

/// CUDA's fusion (the graph compiler's `fusion` pass): the shared fusion with the CuTe kernels' limits (elementwise and
/// GEMM epilogues), with `fuse`.
impl crate::codegen::fusion::Fuses for Cuda {
    /// Runs the shared fusion with the CUDA backend's limits.
    fn fuse(m: &mut DeviceModule<'_, Cuda>) -> Result<(), String> {
        let backend = backend(m);
        m.graph = fuse(&m.graph, &[], &[], &backend, |body, _| named(body));
        Ok(())
    }
}
