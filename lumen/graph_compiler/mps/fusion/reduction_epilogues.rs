use super::analysis::Analysis;
use super::{constant, elementwise};
use crate::graph::Primitive;
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::diamonds::Row;

pub(super) struct ReductionEpilogues {
    pub(super) epilogue: Vec<Option<usize>>,
    pub(super) ends: Vec<Option<usize>>,
    pub(super) inside: Vec<bool>,
}

pub(super) fn reduction_epilogues(
    a: &Analysis,
    rows: &[Row],
    config: &MpsConfig,
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
        if !config.reduction_epilogues || !live[i] || !fusible[i] || !reduction || in_row {
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
            if !fusible[u] || !step || !reads {
                break;
            }
            taken.push(u);
            after[un.output] = true;
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
