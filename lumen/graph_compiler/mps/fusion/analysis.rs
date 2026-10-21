use super::{constant, elementwise, fusible};
use crate::graph::{Graph, Node, Primitive, Var};

pub(super) struct Analysis<'a> {
    pub(super) graph: &'a Graph,
    pub(super) nodes: &'a [Node],
    pub(super) n: usize,
    pub(super) producer: Vec<Option<usize>>,
    pub(super) is_output: Vec<bool>,
    pub(super) live: Vec<bool>,
    pub(super) users: Vec<Vec<usize>>,
    pub(super) fusible: Vec<bool>,
}

impl<'a> Analysis<'a> {
    pub(super) fn new(graph: &'a Graph) -> Self {
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

    pub(super) fn last_read(&self, v: Var) -> Option<usize> {
        self.users[v].iter().copied().max()
    }

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
