//! Slices read only by GEMMs, turned into strided views of their operand instead of copies.

use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::tensor::contiguous_strides;

/// Whether a dot (`Primitive`, its two operands' types) reads operand `k` laid out with `strides`.
pub(crate) type ReadsStrided = fn(&Primitive, [&TensorType; 2], usize, &[usize]) -> bool;

/// Marks each slice `dot_views` accepts as a view.
pub(crate) fn place_views(
    graph: &mut Graph,
    reads: ReadsStrided,
    gemm_dot: fn(&Graph) -> Option<&Node>,
) {
    for v in dot_views(graph, reads, gemm_dot) {
        let node = graph.nodes.iter_mut().find(|n| n.output == v);

        if let Some(Primitive::Slice { view, .. }) = node.map(|n| &mut n.primitive) {
            *view = true;
        }
    }
}

/// Outputs of slices whose readers are all dots or GEMM fusions able to read the sliced input's strides.
fn dot_views(
    graph: &Graph,
    reads: ReadsStrided,
    gemm_dot: fn(&Graph) -> Option<&Node>,
) -> Vec<Var> {
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
            Primitive::Fusion { body, .. } => (0..n.inputs.len()).all(|k| {
                n.inputs[k] != v || fusion_reads_strided(body, k, &strides, reads, gemm_dot)
            }),
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
    gemm_dot: fn(&Graph) -> Option<&Node>,
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
