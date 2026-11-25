//! MPS fusion: the shared fusion (`passes::fusion`) with Metal's kernels' limits and shapes,
//! each fusion compiled to one Metal kernel.

use crate::DType;
use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::passes::{attention, place_views};
pub(crate) use crate::graph_compiler::{elementwise, label};
use crate::plan::StepBase;

/// Whether `node` can live inside a fusion kernel: elementwise ops, gathers and reductions under
/// 2^32 elements, none touching float64.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;

    let loop_op = matches!(
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
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
    );

    let gather = matches!(node.primitive, Gather { .. });

    let reduction = matches!(node.primitive, ReduceSum { .. } | ReduceMax { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;

    let f64 = node
        .inputs
        .iter()
        .chain([&node.output])
        .any(|&v| graph.type_of(v).dtype == DType::F64);

    (loop_op || gather || reduction) && !f64
}

/// Whether the kernel chosen for `body` can write its outputs through strided views.
pub(crate) fn writes_views(body: &Graph) -> bool {
    gemm_dot(body).is_none() && row_reductions(body).is_empty() && reduction_root(body).is_none()
}

/// Whether fusion input `k` may be bound as a strided view: it is not attention and no dot reads it.
pub(crate) fn loads_strided(body: &Graph, k: usize) -> bool {
    let Some(&v) = body.inputs().get(k) else {
        return false;
    };

    let template = |p: &Primitive| matches!(p, Primitive::DotGeneral { .. });

    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && !body
            .nodes()
            .iter()
            .any(|n| template(&n.primitive) && n.inputs.contains(&v))
}

/// Whether some input of `step` is a view its kernel loads element by element (a strided variant of the
/// kernel), which tuned kernels don't; views a GEMM's dot strides through need none.
pub(crate) fn loads_views(step: &StepBase) -> bool {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        return step.views.iter().any(Option::is_some);
    };

    step.views
        .iter()
        .enumerate()
        .any(|(k, view)| view.is_some() && loads_strided(body, k))
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout MPS's matmul
/// kernels stride through: an attention's dots are its kernel's, not a GEMM's.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    attention::of_body(body).is_none()
        && attention::backward_of_body(body).is_none()
        && place_views::fusion_reads_strided(
            body,
            k,
            strides,
            crate::codegen::dot_general::mps::reads_strided,
        )
}
