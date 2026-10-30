//! Per-graph facts shared by the loop fusion stages: producers, users, liveness and fusibility.

use super::{constant, elementwise};
use crate::graph::{Graph, Node, Primitive, Var};

/// Use-def facts about a graph, indexed by value or node.
pub(super) struct Analysis<'a> {
    pub(super) graph: &'a Graph,
    pub(super) nodes: &'a [Node],
    /// Number of values in the graph.
    pub(super) n: usize,
    /// Node producing each value, `None` for graph inputs.
    pub(super) producer: Vec<Option<usize>>,
    /// Whether each value is a graph output.
    pub(super) is_output: Vec<bool>,
    /// Whether each node contributes to an output.
    pub(super) live: Vec<bool>,
    /// Live nodes reading each value, in increasing order without repeats.
    pub(super) users: Vec<Vec<usize>>,
    /// Whether each node may join a fusion; in-place concatenations may not.
    pub(super) fusible: Vec<bool>,
}

impl<'a> Analysis<'a> {
    /// Computes the analysis of `graph`, `fusible` telling which nodes may join a fusion.
    pub(super) fn new(graph: &'a Graph, fusible: fn(&Graph, &Node) -> bool) -> Self {
        let nodes = graph.nodes();
        let n = graph.types.len();
        let mut producer: Vec<Option<usize>> = vec![None; n];

        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
        }

        let mut is_output = vec![false; n];

        for &v in graph.outputs() {
            is_output[v] = true;
        }

        let mut live = vec![false; nodes.len()];
        let mut live_var = is_output.clone();

        for (i, node) in nodes.iter().enumerate().rev() {
            if live_var[node.output] {
                live[i] = true;

                for &v in &node.inputs {
                    live_var[v] = true;
                }
            }
        }

        let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| live[i]) {
            for &v in &node.inputs {
                if users[v].last() != Some(&i) {
                    users[v].push(i);
                }
            }
        }

        let in_place = |node: &Node| {
            matches!(
                node.primitive,
                Primitive::Concatenate { in_place: true, .. }
            )
        };

        let fusible: Vec<bool> = nodes
            .iter()
            .map(|node| fusible(graph, node) && !in_place(node))
            .collect();

        Analysis {
            graph,
            nodes,
            n,
            producer,
            is_output,
            live,
            users,
            fusible,
        }
    }

    /// Whether `v` depends on node `k`, searching only producers at or after `k`.
    pub(super) fn depends(&self, v: Var, k: usize) -> bool {
        let (nodes, producer) = (self.nodes, &self.producer);
        let (mut seen, mut stack) = (vec![false; nodes.len()], vec![v]);

        while let Some(v) = stack.pop() {
            match producer[v].filter(|&p| p >= k && !seen[p]) {
                Some(p) if p == k => return true,
                Some(p) => {
                    seen[p] = true;
                    stack.extend(&nodes[p].inputs);
                }
                None => {}
            }
        }

        false
    }

    /// Index of the last live node reading `v`.
    pub(super) fn last_read(&self, v: Var) -> Option<usize> {
        self.users[v].iter().copied().max()
    }

    /// Whether elementwise node `i` is cheap to recompute in each reader instead of being stored,
    /// because every input (looking through broadcasts and reshapes) stays alive until its last reader anyway.
    pub(super) fn recomputed(&self, i: usize) -> bool {
        let (graph, nodes, producer, is_output) =
            (self.graph, self.nodes, &self.producer, &self.is_output);

        let last_read = |v: Var| self.last_read(v);
        let node = &nodes[i];

        let Some(end) = last_read(node.output) else {
            return false;
        };

        let alive = |v: Var| {
            let mut v = v;

            loop {
                if producer[v].is_none()
                    || constant(graph, producer, v)
                    || is_output[v]
                    || last_read(v).is_some_and(|r| r >= end)
                {
                    return true;
                }

                match producer[v].filter(|&p| {
                    matches!(
                        nodes[p].primitive,
                        Primitive::BroadcastInDim { .. } | Primitive::Reshape { .. }
                    )
                }) {
                    Some(p) => v = nodes[p].inputs[0],
                    None => return false,
                }
            }
        };

        elementwise(&node.primitive)
            && !is_output[node.output]
            && node.inputs.iter().all(|&v| alive(v))
    }
}
