//! Triton fusion: the shared fusion (`passes::fusion`, as MPS runs it) with the Triton
//! kernels' limits and shapes, and which nodes left outside any fusion need one-node kernels.

use crate::codegen::devices::triton::triton_type;
use crate::codegen::kinds::{combine_root, gemm_dot, reduction_root, row_reductions};
use crate::graph::{Graph, Node, Primitive, Var};

/// Whether `node` can be emitted inside a Triton fusion kernel: a supported primitive whose
/// inputs and output all have Triton dtypes.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;

    let op = matches!(
        node.primitive,
        Add | Sub
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
            | Unary(_)
            | Binary(_)
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | SwapPairs { .. }
            | Slice { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
            | ReduceSum { .. }
            | ReduceMax { .. }
            | DynamicSlice { .. }
            | Gather { .. }
            | Concatenate { .. }
    );

    let typed = |v: &Var| triton_type(graph.type_of(*v).dtype).is_some();
    op && node.inputs.iter().chain([&node.output]).all(typed)
}

/// Name of a fusion of a GEMM and the normalization row reading its product.
const DOT_ROW: &str = "dot_row";

/// Name of a normalization row fusion.
const ROW: &str = "row";

/// Name of a GEMM fusion with its epilogue.
const DOT_GENERAL: &str = "dot_general";

/// Name of a reduction fusion with its prologue and epilogue.
const REDUCTION: &str = "reduction";

/// Name of a split-K combine running the epilogue its GEMM would unsplit (`kinds::combine_root`).
pub(crate) const COMBINE: &str = "combine";

/// Name of an elementwise fusion.
const ELEMENTWISE: &str = "elementwise";

/// The name of a fused `body`, by its kind as MPS's codegen reads it (`codegen::kinds`): a GEMM and the
/// row reading it, a normalization row, a GEMM with its epilogue, a split-K combine with the epilogue its GEMM
/// would run, a reduction with its prologue and epilogue, or an elementwise fusion.
pub(crate) fn named(body: &Graph) -> String {
    let row = !row_reductions(body).is_empty();

    match (gemm_dot(body).is_some(), row) {
        (true, true) => DOT_ROW,
        (false, true) => ROW,
        (true, false) => DOT_GENERAL,
        _ if combine_root(body).is_some() => COMBINE,
        _ if reduction_root(body).is_some() => REDUCTION,
        _ => ELEMENTWISE,
    }
    .into()
}
