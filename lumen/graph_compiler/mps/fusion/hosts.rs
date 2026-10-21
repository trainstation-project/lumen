use super::analysis::Analysis;
use super::dot_epilogues::DotEpilogues;
use super::reduction_epilogues::ReductionEpilogues;
use super::{elementwise, expensive, is_reduction};
use crate::graph::{Graph, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::diamonds::Row;

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

pub(super) fn hosts(
    a: &Analysis,
    rows: &[Row],
    config: &CompilerConfig,
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
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);
    let mut host: Vec<Option<usize>> = dot_aux;
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = config.multi_output_fusion
            && live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && (is_output[node.output] || expensive(graph, node))
            && dot_of[i].is_none()
            && !rows.iter().any(|r| r.root == i && !r.outputs.is_empty());
        let Some(&first) = users[node.output].first().filter(|_| candidate) else {
            continue;
        };
        let reader = first;
        let mut first = first;
        while !root[first] && users[nodes[first].output].len() == 1 {
            first = users[nodes[first].output][0];
        }
        let fusion = ends[first].unwrap_or(first);
        let after = users[node.output]
            .iter()
            .all(|&u| u == reader || u > fusion);
        if root[fusion]
            && after
            && fusible[first]
            && !row_root(first)
            && dot_of[fusion].is_none()
            && host[fusion].is_none()
            && at_index(graph, producer, root, first, node.output)
        {
            host[i] = Some(fusion);
        }
    }

    for row in rows {
        for &k in row.outputs.iter().chain(&row.partials) {
            host[k] = Some(row.root);
        }
    }
    host
}

fn at_index(graph: &Graph, producer: &[Option<usize>], root: &[bool], r: usize, v: Var) -> bool {
    let nodes = graph.nodes();
    let start = match is_reduction(&nodes[r]) {
        true => nodes[r].inputs[0],
        false => nodes[r].output,
    };
    let mut stack = vec![start];
    while let Some(x) = stack.pop() {
        if x == v {
            return true;
        }
        let Some(p) = producer[x] else { continue };
        if (p == r || !root[p]) && elementwise(&nodes[p].primitive) {
            stack.extend(&nodes[p].inputs);
        }
    }
    false
}
