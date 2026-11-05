//! Slices read only by GEMMs, turned into strided views of their operand instead of copies.

use crate::codegen::kinds::gemm_dot;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::tensor::contiguous_strides;

/// Whether a dot (`Primitive`, its two operands' types) reads operand `k` laid out with `strides`.
pub(crate) type ReadsStrided = fn(&Primitive, [&TensorType; 2], usize, &[usize]) -> bool;

/// Marks each slice `dot_views` accepts as a view.
pub(crate) fn place_views(graph: &mut Graph, reads: ReadsStrided) {
    for v in dot_views(graph, reads) {
        let node = graph.nodes.iter_mut().find(|n| n.output == v);

        if let Some(Primitive::Slice { view, .. }) = node.map(|n| &mut n.primitive) {
            *view = true;
        }
    }
}

/// Outputs of slices whose readers are all dots or GEMM fusions able to read the sliced input's strides.
fn dot_views(graph: &Graph, reads: ReadsStrided) -> Vec<Var> {
    let nodes = graph.nodes();
    let mut views = Vec::new();

    for node in nodes {
        let (Primitive::Slice { .. }, &[x]) = (&node.primitive, node.inputs.as_slice()) else {
            continue;
        };

        let v = node.output;

        if graph.outputs().contains(&v) || views.contains(&x) {
            continue;
        }

        let strides = contiguous_strides(&graph.type_of(x).shape);
        let mut readers = nodes.iter().filter(|n| n.inputs.contains(&v)).peekable();
        let read = readers.peek().is_some();

        let in_place = readers.all(|n| match &n.primitive {
            Primitive::DotGeneral { .. } => {
                let operands = [graph.type_of(n.inputs[0]), graph.type_of(n.inputs[1])];
                (0..2).all(|k| n.inputs[k] != v || reads(&n.primitive, operands, k, &strides))
            }
            Primitive::Fusion { body, .. } => (0..n.inputs.len())
                .all(|k| n.inputs[k] != v || fusion_reads_strided(body, k, &strides, reads)),
            _ => false,
        });

        if read && in_place {
            views.push(v);
        }
    }

    views
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout it can stride through.
pub(crate) fn fusion_reads_strided(
    body: &Graph,
    k: usize,
    strides: &[usize],
    reads: ReadsStrided,
) -> bool {
    let (Some(dot), Some(&v)) = (gemm_dot(body), body.inputs().get(k)) else {
        return false;
    };

    let side = dot.inputs.iter().position(|&u| u == v);

    let only = body
        .nodes()
        .iter()
        .all(|n| std::ptr::eq(n, dot) || !n.inputs.contains(&v));

    let operands = [body.type_of(dot.inputs[0]), body.type_of(dot.inputs[1])];

    match side {
        Some(side) if only && dot.inputs[0] != dot.inputs[1] => {
            reads(&dot.primitive, operands, side, strides)
        }
        _ => false,
    }
}

/// Compiler pass marking each slice read only by GEMMs that stride through it (`reads`, the backend's
/// matmul kernels' rule) a view.
pub(crate) struct PlaceViews {
    pub reads: ReadsStrided,
}

impl<M: Module> CompilerPass<M> for PlaceViews {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "place_views"
    }

    /// Marks the slices `place_views` accepts.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        place_views(m.graph_mut(), self.reads);
        Ok(())
    }
}
