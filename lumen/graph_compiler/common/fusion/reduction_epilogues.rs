//! Finds elementwise epilogues that can run inside the kernel of the reduction feeding them.

use super::analysis::Analysis;
use super::{Backend, constant, elementwise};
use crate::graph::Primitive;
use crate::graph_compiler::common::diamonds::Row;

/// Reduction epilogue assignment, indexed by node.
pub(super) struct ReductionEpilogues {
    /// For the last node of an epilogue, the reduction it belongs to.
    pub(super) epilogue: Vec<Option<usize>>,
    /// For a reduction, the last node of its epilogue.
    pub(super) ends: Vec<Option<usize>>,
    /// Whether a node sits strictly inside some reduction's epilogue.
    pub(super) inside: Vec<bool>,
}

/// For each fusible reduction outside normalization rows, takes the longest following chain of
/// elementwise ops and reshapes over its result and constants whose intermediates nobody else reads.
pub(super) fn reduction_epilogues(
    a: &Analysis,
    rows: &[Row],
    backend: &Backend,
) -> ReductionEpilogues {
    let (graph, nodes, n) = (a.graph, a.nodes, a.n);

    let (producer, is_output, live, users, fusible) =
        (&a.producer, &a.is_output, &a.live, &a.users, &a.fusible);

    let mut epilogue: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut ends: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_reduction_epilogue = vec![false; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        let reduction = matches!(
            node.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        );

        let in_row = rows.iter().any(|r| r.reductions.contains(&i));

        let of_row = |u: usize| {
            rows.iter()
                .any(|r| r.root == u || r.inner.contains(&u) || r.reductions.contains(&u))
        };

        if !backend.reduction_epilogues || !live[i] || !fusible[i] || !reduction || in_row {
            continue;
        }

        let (mut taken, mut after, mut end) = (vec![i], vec![false; n], None);
        after[node.output] = true;

        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }

            let step =
                elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });

            let reads = un
                .inputs
                .iter()
                .all(|&x| after[x] || constant(graph, producer, x));

            if !fusible[u] || !step || !reads || of_row(u) {
                break;
            }

            taken.push(u);
            after[un.output] = true;

            // The chain may end here only if every earlier member is read solely within the chain.
            let inside = taken[..taken.len() - 1].iter().all(|&t| {
                let v = nodes[t].output;
                !is_output[v] && users[v].iter().all(|r| taken.contains(r))
            });

            if inside && elementwise(&un.primitive) {
                end = Some(u);
            }
        }

        if let Some(end) = end {
            (epilogue[end], ends[i]) = (Some(i), Some(end));

            for &t in taken.iter().filter(|&&t| t != i && t < end) {
                in_reduction_epilogue[t] = true;
            }
        }
    }

    ReductionEpilogues {
        epilogue,
        ends,
        inside: in_reduction_epilogue,
    }
}
