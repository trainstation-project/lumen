//! The slices dots read in place: strided views of the sliced value,
//! marked so in the graph ([`Primitive::Slice`]'s `view`,
//! [`Graph::placements`]): no step computes them.

use super::{MpsModule, codegen};
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::CompilerPass;
use crate::ops::dot_general::mps::reads_strided;
use crate::tensor::contiguous_strides;

/// The slices its dots read in place ([`dot_views`]): marked so.
pub(super) struct PlaceViews;

impl<'a> CompilerPass<MpsModule<'a>> for PlaceViews {
    fn name(&self) -> &'static str {
        "place_views"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        for v in dot_views(&m.graph) {
            let node = m.graph.nodes.iter_mut().find(|n| n.output == v);
            if let Some(Primitive::Slice { view, .. }) = node.map(|n| &mut n.primitive) {
                *view = true;
            }
        }
        Ok(())
    }
}

/// The slices of `graph` its dots read in place, as strided views of the
/// sliced value (marked so, `Slice`'s `view`; the matmul kernels take any
/// operand strides): those read by dots alone (or contractions with their
/// epilogues, [`fusion_reads_strided`]), each of which reads it in matmul
/// form at those strides, that are not outputs or of a value that is such
/// a view itself.
fn dot_views(graph: &Graph) -> Vec<Var> {
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
                (0..2)
                    .all(|k| n.inputs[k] != v || reads_strided(&n.primitive, operands, k, &strides))
            }
            Primitive::Fusion { body, .. } => (0..n.inputs.len())
                .all(|k| n.inputs[k] != v || fusion_reads_strided(body, k, &strides)),
            _ => false,
        });
        if read && in_place {
            views.push(v);
        }
    }
    views
}

/// Whether a fusion with `body` reads its input `k` in place at `strides`
/// (in elements): a contraction with its epilogue, whose dot alone reads it,
/// in matmul form at them.
pub(crate) fn fusion_reads_strided(body: &Graph, k: usize, strides: &[usize]) -> bool {
    let (Some(dot), Some(&v)) = (codegen::gemm_dot(body), body.inputs().get(k)) else {
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
            crate::ops::dot_general::mps::reads_strided(&dot.primitive, operands, side, strides)
        }
        _ => false,
    }
}
