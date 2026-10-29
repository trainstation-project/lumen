//! Triton loop fusion: the shared loop fusion (`common::fusion`, as MPS runs it) with the Triton
//! kernels' limits and shapes, and the nodes left outside any fusion wrapped in one-node kernels.

use super::TritonModule;
use crate::codegen::kinds::row_reductions;
use crate::codegen::regions::expansion;
use crate::codegen::triton::{self as codegen, dot_row_fits, fusible};
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::fusion::{Backend, fuse};
use crate::graph_compiler::common::place_views::place_views;
use crate::graph_compiler::pass::CompilerPass;

/// Name of a fusion of a GEMM and the normalization row reading its product.
const DOT_ROW: &str = "dot_row";

/// Name of a normalization row fusion.
const ROW: &str = "row";

/// Name of every other fusion: an elementwise loop, a reduction and its epilogue, a GEMM and its epilogue.
const LOOP: &str = "loop";

/// The shared loop fusion's view of Triton: kernels take any number of tensors, a GEMM epilogue
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

/// The name Triton's runner dispatches a fused `body` on, by its kind as MPS's codegen reads it
/// (`codegen::kinds`): a GEMM and the row reading it, a normalization row, or a loop (an elementwise
/// loop, a reduction with its epilogue, a GEMM with its epilogue).
fn named(body: &Graph) -> String {
    let row = !row_reductions(body).is_empty();
    match (codegen::gemm_dot(body).is_some(), row) {
        (true, true) => DOT_ROW,
        (false, true) => ROW,
        _ => LOOP,
    }
    .into()
}

/// Compiler pass running the shared loop fusion for Triton.
pub(super) struct LoopFusion;

impl CompilerPass<TritonModule<'_>> for LoopFusion {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "loop_fusion"
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

/// Compiler pass wrapping each node left outside any fusion that a Triton loop kernel computes in a
/// fusion of its own: Triton has no other kernel for it.
pub(super) struct Alone;

impl CompilerPass<TritonModule<'_>> for Alone {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "alone"
    }

    /// Wraps every lone fusible node but views of their operand (reshapes, slices placed as views) and
    /// in-place concatenations.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        m.graph = wrapped(&m.graph, |graph, n| {
            fusible(graph, n)
                && !matches!(
                    n.primitive,
                    Primitive::Reshape { .. }
                        | Primitive::Slice { view: true, .. }
                        | Primitive::Concatenate { in_place: true, .. }
                )
        });
        Ok(())
    }
}

/// `graph` with each node `wraps` accepts replaced by a fusion whose body is that node alone, its
/// operands read once each.
fn wrapped(graph: &Graph, wraps: impl Fn(&Graph, &Node) -> bool) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        if !wraps(graph, node) {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            out.set_label(map[node.output], node.label);
            continue;
        }
        let mut reads: Vec<Var> = Vec::new();
        for &v in &node.inputs {
            if !reads.contains(&v) {
                reads.push(v);
            }
        }
        let mut body = Graph::new();
        let ins: Vec<Var> = reads
            .iter()
            .map(|&v| body.input(graph.type_of(v).clone()))
            .collect();
        let operands: Vec<Var> = node
            .inputs
            .iter()
            .map(|v| ins[reads.iter().position(|r| r == v).expect("a read")])
            .collect();
        let value = body
            .apply(node.primitive.clone(), &operands)
            .expect("the node alone");
        body.set_outputs(&[value]).expect("its value");
        let name = named(&body);
        let label = graph.label(node);
        let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(Primitive::Fusion { name, label, body }, &reads)
            .expect("a fusion typed as the node");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
