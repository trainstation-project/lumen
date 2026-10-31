//! Triton fusion: the shared fusion (`common::fusion`, as MPS runs it) with the Triton
//! kernels' limits and shapes, and which nodes left outside any fusion need one-node kernels.

use super::TritonModule;
use crate::codegen::kinds::row_reductions;
use crate::codegen::regions::expansion;
use crate::codegen::triton::{self as codegen, dot_row_fits, fusible};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::alone::Lone;
use crate::graph_compiler::common::fusion::{Backend, fuse};
use crate::graph_compiler::common::place_views::place_views;
use crate::graph_compiler::pass::CompilerPass;

/// Name of a fusion of a GEMM and the normalization row reading its product.
const DOT_ROW: &str = "dot_row";

/// Name of a normalization row fusion.
const ROW: &str = "row";

/// Name of a GEMM fusion with its epilogue.
const DOT_GENERAL: &str = "dot_general";

/// Name of a reduction fusion with its prologue and epilogue.
const REDUCTION: &str = "reduction";

/// Name of an elementwise fusion.
const ELEMENTWISE: &str = "elementwise";

/// The shared fusion's view of Triton: kernels take any number of tensors, a GEMM epilogue
/// writes at most 4 extra outputs and its two results into a concatenation's slots (as on MPS), and
/// a GEMM joins the row reading it when one tile holds whole rows (`dot_row_fits`).
fn backend(m: &TritonModule<'_>) -> Backend {
    let config = &m.options.config.cuda;

    Backend {
        fusible,
        max_inputs: usize::MAX,
        max_aux: 4,
        multi_output_fusion: config.multi_output_fusion,
        reduction_epilogues: true,
        contraction_epilogues: true,
        expansion: |graph, dot, concat| expansion(graph, dot, concat).is_some(),
        row_dot: dot_row_fits,
    }
}

/// The name of a fused `body`, by its kind as MPS's codegen reads it (`codegen::kinds`): a GEMM and the
/// row reading it, a normalization row, a GEMM with its epilogue, a reduction with its prologue and
/// epilogue, or an elementwise fusion.
fn named(body: &Graph) -> String {
    let row = !row_reductions(body).is_empty();

    match (codegen::gemm_dot(body).is_some(), row) {
        (true, true) => DOT_ROW,
        (false, true) => ROW,
        (true, false) => DOT_GENERAL,
        _ if codegen::reduction_root(body).is_some() => REDUCTION,
        _ => ELEMENTWISE,
    }
    .into()
}

/// Compiler pass running the shared fusion for Triton.
pub(super) struct Fusion;

impl CompilerPass<TritonModule<'_>> for Fusion {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "fusion"
    }

    /// Fuses the graph around the normalization rows the diamonds pass found.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        let scalars: Vec<Var> = super::super::positions(&m.options.scalars)
            .into_iter()
            .map(|i| m.graph.inputs()[i])
            .collect();

        let backend = backend(m);
        m.graph = fuse(&m.graph, &m.rows, &scalars, &backend, |body, _| named(body));
        m.rows.clear();
        Ok(())
    }
}

/// Compiler pass marking slices read only by GEMMs as strided views, as on MPS: Triton's GEMM reads its
/// operands at any strides.
pub(super) struct PlaceViews;

impl CompilerPass<TritonModule<'_>> for PlaceViews {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "place_views"
    }

    /// Marks each slice read only by GEMMs a view (`common::place_views`).
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        place_views(&mut m.graph, |_, _, _, _| true, codegen::gemm_dot);
        Ok(())
    }
}

impl Lone for TritonModule<'_> {
    /// Whether lone `n` is one a Triton loop kernel computes (Triton has no other kernel for it): every
    /// fusible node but views of their operand (reshapes, slices placed as views) and in-place
    /// concatenations.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n)
            && !matches!(
                n.primitive,
                Primitive::Reshape { .. }
                    | Primitive::Slice { view: true, .. }
                    | Primitive::Concatenate { in_place: true, .. }
            )
    }

    /// Triton passes no inputs by value.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Names the kernel by the body's kind.
    fn kernel(&mut self, body: &Graph, _: &[bool]) -> String {
        named(body)
    }
}
