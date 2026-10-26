//! Picks fusion roots and the hosts that absorb other fusions' values as extra outputs.

use super::analysis::Analysis;
use super::dot_epilogues::DotEpilogues;
use super::reduction_epilogues::ReductionEpilogues;
use super::{expensive, is_reduction};
use crate::graph_compiler::MpsConfig;
use crate::graph_compiler::common::diamonds::Row;
use crate::graph_compiler::common::multi_output::Readers;

/// Whether each node ends a fusion and must be materialized: unfusible nodes, reductions outside
/// epilogues, epilogue ends, row roots, outputs, values read by unfusible nodes and shared expensive ops.
pub(super) fn roots(
    a: &Analysis,
    rows: &[Row],
    reductions: &ReductionEpilogues,
    dots: &DotEpilogues,
) -> Vec<bool> {
    let (graph, nodes, is_output, users, fusible) =
        (a.graph, a.nodes, &a.is_output, &a.users, &a.fusible);
    let (dot_end, dot_of, in_epilogue) = (&dots.end, &dots.of, &dots.inside);
    let (epilogue, ends, in_reduction_epilogue) =
        (&reductions.epilogue, &reductions.ends, &reductions.inside);
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let users = &users[node.output];
            if dot_end[i].is_some() || in_epilogue[i] || in_reduction_epilogue[i] {
                return false;
            }
            if rows.iter().any(|r| {
                r.inner.contains(&i) || r.reductions.contains(&i) || r.partials.contains(&i)
            }) {
                return false;
            }
            let inside = ends[i].is_some();
            !fusible[i]
                || (is_reduction(node) && !inside)
                || epilogue[i].is_some()
                || dot_of[i].is_some()
                || row_root(i)
                || is_output[node.output]
                || users.iter().any(|&u| !fusible[u])
                || (expensive(graph, node) && users.len() > 1)
        })
        .collect()
}

/// For each root, a later fusion that can also write its value as an extra output (multi-output fusion),
/// plus GEMM auxiliary outputs and row outputs and partials hosted by their row.
pub(super) fn hosts(
    a: &Analysis,
    rows: &[Row],
    config: &MpsConfig,
    root: &[bool],
    reductions: &ReductionEpilogues,
    dots: &DotEpilogues,
) -> Vec<Option<usize>> {
    let (graph, nodes, producer, is_output, live, users, fusible) = (
        a.graph,
        a.nodes,
        &a.producer,
        &a.is_output,
        &a.live,
        &a.users,
        &a.fusible,
    );
    let (dot_of, dot_aux, ends) = (&dots.of, dots.aux.clone(), &reductions.ends);
    let epilogue = &reductions.epilogue;
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);
    let mut host: Vec<Option<usize>> = dot_aux;
    let readers = Readers {
        graph,
        producer,
        users,
    };
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = config.multi_output_fusion
            && live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && (is_output[node.output] || expensive(graph, node))
            && dot_of[i].is_none()
            && epilogue[i].is_none()
            && !rows.iter().any(|r| r.root == i && !r.outputs.is_empty());
        if !candidate {
            continue;
        }
        // Follow single-reader non-roots to the fusion that will consume this value.
        let fusion_of = |reader: usize| {
            let mut first = reader;
            while !root[first] && users[nodes[first].output].len() == 1 {
                first = users[nodes[first].output][0];
            }
            (first, ends[first].unwrap_or(first))
        };
        let member = |first: usize, p: usize| p == first || !root[p];
        let hostable = |first: usize, fusion: usize| {
            root[fusion]
                && fusible[first]
                && !row_root(first)
                && dot_of[fusion].is_none()
                && host[fusion].is_none()
        };
        host[i] = readers.host(i, fusion_of, member, hostable);
    }

    for row in rows {
        for &k in row.outputs.iter().chain(&row.partials) {
            host[k] = Some(row.root);
        }
    }
    host
}
