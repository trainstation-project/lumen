//! Triton fusion: the shared fusion (`passes::fusion`, as MPS runs it) with the Triton
//! kernels' limits and shapes, and which nodes left outside any fusion need one-node kernels.

use crate::codegen::devices::triton::triton_type;
use crate::codegen::dot_general::triton::gemv_body;
use crate::codegen::dot_general::triton::row::dot_row_fits;
use crate::codegen::kinds::{combine_root, gemm_dot, reduction_root, row_reductions};
use crate::codegen::regions::expansion;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::device::{DeviceModule, Triton};
use crate::graph_compiler::passes::fusion::{Backend, fuse};

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

/// The shared fusion's view of Triton: kernels take any number of tensors, a GEMM epilogue
/// writes at most 4 extra outputs and its two results into a concatenation's slots (as on MPS), and
/// a GEMM joins the row reading it when one tile holds whole rows (`dot_row_fits`).
fn backend(m: &DeviceModule<'_, Triton>) -> Backend {
    let config = &m.options.config;

    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: true,
        contraction_epilogues: true,
        dot_epilogue: |_, _| true,
        expansion: |graph, dot, concat| expansion(graph, dot, concat).is_some(),
        // A matrix-vector product runs on the GEMV kernel, not a row kernel's GEMM.
        row_dot: |body| dot_row_fits(body) && !gemv_body(body),
        swap_epilogues: true,
        combine_epilogues: true,
        update_epilogues: true,
    }
}

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

/// Triton's fusion (the graph compiler's `fusion` pass): the shared fusion with Triton's limits, each fusion named by
/// its kind (`named`), with `fuse`.
impl crate::codegen::fusion::Fuses for Triton {
    /// Fuses the graph around the normalization rows the diamonds pass found.
    fn fuse(m: &mut DeviceModule<'_, Triton>) -> Result<(), String> {
        let scalars: Vec<Var> = crate::graph_compiler::positions(&m.options.scalars)
            .into_iter()
            .map(|i| m.graph.inputs()[i])
            .collect();

        let backend = backend(m);
        m.graph = fuse(&m.graph, &m.rows, &scalars, &backend, |body, _| named(body));
        m.rows.clear();
        Ok(())
    }
}

/// Most tensors one horizontally fused Triton kernel may read and write.
pub(crate) const HORIZONTAL_BUFFERS: usize = 64;

/// Whether horizontal fusion may merge `node`: an `elementwise` fusion without reductions or matmuls, as
/// MPS's are; a node alone joins only when elementwise (MPS runs a lone fill, iota or the like as its own
/// step).
pub(crate) fn horizontal_candidate(_: &Graph, node: &Node) -> bool {
    let Primitive::Fusion { name, body, .. } = &node.primitive else {
        return false;
    };

    let alone = match body.nodes() {
        [n] => !crate::graph_compiler::elementwise(&n.primitive),
        _ => false,
    };

    name == "elementwise" && !alone && reduction_root(body).is_none() && gemm_dot(body).is_none()
}

/// Whether `producer` ends a kernel of its own writing fresh memory, so it can write a concatenation's operand
/// straight into its slot through the slot's strides: a matmul, or a fusible elementwise or copying layout op
/// (a reshape or view slice aliases its operand instead). Fusion leaves such a concatenate unfused, so each
/// producer is its kernel's root.
pub(crate) fn slot_writes(graph: &Graph, producer: &Node) -> bool {
    let copies = matches!(
        producer.primitive,
        Primitive::Transpose { .. }
            | Primitive::BroadcastInDim { .. }
            | Primitive::Slice { view: false, .. }
    );

    let kernel = fusible(graph, producer)
        && (crate::graph_compiler::elementwise(&producer.primitive) || copies);

    kernel || matches!(producer.primitive, Primitive::DotGeneral { .. })
}

/// Triton merges, with `fuse`, non-trivial `elementwise` fusions without reductions or matmuls, grouped by tiling.
impl crate::codegen::fusion::HorizontallyFuses for Triton {
    const GROUPING: crate::graph_compiler::passes::horizontal::Grouping =
        crate::graph_compiler::passes::horizontal::Grouping::Tiling;
    const BUFFERS: usize = crate::codegen::fusion::triton::HORIZONTAL_BUFFERS;

    /// `fusion::triton::horizontal_candidate`.
    fn candidate(graph: &Graph, node: &Node) -> bool {
        crate::codegen::fusion::triton::horizontal_candidate(graph, node)
    }
}

/// Triton's kernels writing a concatenation's operands into their slots, with `fuse`: kernels of their own writing
/// fresh memory (`fusion::triton::slot_writes`).
impl crate::codegen::fusion::Slots for DeviceModule<'_, Triton> {
    /// What `fusion::triton::slot_writes` accepts.
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        crate::codegen::fusion::triton::slot_writes(&self.graph, producer)
    }
}
