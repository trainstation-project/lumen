//! The kinds of fused body the GPU backends' code generators tell apart, read from the body alone: a
//! GEMM, a reduction with its prologue and epilogue, or a normalization row (reductions over the last
//! axis whose results feed back into their rows), and the column partials riding along a row.

use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::{constant, elementwise};

/// The body's `DotGeneral` node, if any.
pub(crate) fn gemm_dot(body: &Graph) -> Option<&Node> {
    body.nodes()
        .iter()
        .find(|n| matches!(n.primitive, Primitive::DotGeneral { .. }))
}

/// The body's only reduction when the first output reaches it through elementwise or reshape
/// nodes alone, else `None`.
pub(crate) fn reduction_root(body: &Graph) -> Option<&Node> {
    use Primitive::*;
    let nodes = body.nodes();
    let mut producer = vec![None; body.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    // Walk back from the first output; any non-elementwise node other than the one reduction
    // rules the body out.
    let (mut found, mut stack) = (None, vec![body.outputs()[0]]);

    while let Some(v) = stack.pop() {
        let i = producer[v]?;
        let node = &nodes[i];

        match &node.primitive {
            ReduceSum { .. } | ReduceMax { .. } if found.is_none_or(|f| f == i) => found = Some(i),
            p if elementwise(p) || matches!(p, Reshape { .. }) => stack.extend(
                node.inputs
                    .iter()
                    .filter(|&&u| !constant(body, &producer, u)),
            ),
            _ => return None,
        }
    }

    let reductions = nodes
        .iter()
        .filter(|n| matches!(n.primitive, ReduceSum { .. } | ReduceMax { .. }))
        .count();

    found.filter(|_| reductions == 1).map(|i| &nodes[i])
}

/// The body's split-K combine: its only reduction, a sum over axis 0 (at most `MAX_SPLIT` K chunks) of a
/// body input (the chunks' products), which every body output is computed from; else `None`.
pub(crate) fn combine_root(body: &Graph) -> Option<&Node> {
    use crate::graph_compiler::passes::dot::split_k::MAX_SPLIT;

    let mut reductions = body.nodes().iter().filter(|n| {
        matches!(
            n.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    });

    let root = match (reductions.next(), reductions.next()) {
        (Some(root), None) => root,
        _ => return None,
    };

    let Primitive::ReduceSum { axes, .. } = &root.primitive else {
        return None;
    };

    let products = body.type_of(root.inputs[0]);

    // Every output is the sum or computed from it, through the body's nodes.
    let mut from = vec![false; body.types.len()];
    from[root.output] = true;

    for node in body.nodes() {
        from[node.output] |= node.inputs.iter().any(|&v| from[v]);
    }

    (*axes == [0]
        && body.inputs().contains(&root.inputs[0])
        && products.shape.len() >= 2
        && products.shape[0] <= MAX_SPLIT
        && body.outputs().iter().all(|&v| from[v]))
    .then_some(root)
}

/// Body reductions over the last axis, excluding the first output; empty for plain reductions.
pub(crate) fn row_reductions(body: &Graph) -> Vec<&Node> {
    reductions_of_rows(body, true)
}

/// Body reductions that keep the last axis (per-column partial sums written per row group),
/// present only alongside row reductions.
pub(crate) fn row_partials(body: &Graph) -> Vec<&Node> {
    match row_reductions(body).is_empty() {
        true => Vec::new(),
        false => reductions_of_rows(body, false),
    }
}

/// Body reductions other than the first output that include (`last`) or exclude the last axis.
fn reductions_of_rows(body: &Graph, last: bool) -> Vec<&Node> {
    if reduction_root(body).is_some() {
        return Vec::new();
    }

    let out = body.outputs()[0];

    body.nodes()
        .iter()
        .filter(|n| n.output != out)
        .filter(|n| match &n.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                let rank = body.type_of(n.inputs[0]).shape.len();
                axes.contains(&(rank - 1)) == last
            }
            _ => false,
        })
        .collect()
}
