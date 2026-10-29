//! Node schedulers that reorder a graph's nodes to keep less memory live.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{Graph, Primitive, Var};

/// Node-level dependency edges of a graph.
struct Deps {
    /// Per node, the sorted nodes it must run after.
    operands: Vec<Vec<usize>>,
    /// Per node, the nodes that must run after it.
    users: Vec<Vec<usize>>,
}

/// Dependencies of each node on the producers of its inputs, with custom calls also chained in
/// their original order since they may have side effects, and collectives in theirs, which every process
/// must issue in the same order (a process's sends and receives pair with another's in program order).
fn deps(graph: &Graph) -> Deps {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut operands: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut last_custom = None;
    let mut last_collective = None;
    for (i, node) in nodes.iter().enumerate() {
        let mut ops: Vec<usize> = node.inputs.iter().filter_map(|&v| producer[v]).collect();
        if let Primitive::CustomCall { .. } = node.primitive {
            ops.extend(last_custom);
            last_custom = Some(i);
        }
        if node.primitive.is_collective() {
            ops.extend(last_collective);
            last_collective = Some(i);
        }
        ops.sort_unstable();
        ops.dedup();
        operands[i] = ops;
    }
    let mut users = vec![Vec::new(); nodes.len()];
    for (i, ops) in operands.iter().enumerate() {
        for &o in ops {
            users[o].push(i);
        }
    }
    Deps { operands, users }
}

/// Per value, the value whose storage it aliases through chains of reshapes (itself otherwise).
fn roots(graph: &Graph) -> Vec<Var> {
    let mut root: Vec<Var> = (0..graph.types.len()).collect();
    for node in graph.nodes() {
        if let Primitive::Reshape { .. } = node.primitive {
            root[node.output] = root[node.inputs[0]];
        }
    }
    root
}

/// Size of value `v` in bytes.
fn bytes(graph: &Graph, v: Var) -> i64 {
    let ty = graph.type_of(v);
    (ty.numel() * ty.dtype.size_of()) as i64
}

/// Whether node `i` allocates nothing worth tracking: an alias, a fusion output or a scalar.
fn free(graph: &Graph, i: usize) -> bool {
    let node = &graph.nodes()[i];
    matches!(
        node.primitive,
        Primitive::Reshape { .. } | Primitive::FusionOutput { .. }
    ) || graph.type_of(node.output).numel() <= 1
}

/// Greedy list schedule: of the ready nodes, runs the one freeing the most bytes net of its output
/// (free nodes first, then more users, then lower index); returns node indices in run order.
pub(crate) fn list(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    let Deps { operands, users } = deps(graph);
    let root = roots(graph);
    // Graph inputs and outputs live for the whole run, so reading them last frees nothing.
    let mut pinned = vec![false; graph.types.len()];
    for &v in graph.inputs().iter().chain(graph.outputs()) {
        pinned[root[v]] = true;
    }
    let reads: Vec<Vec<Var>> = nodes
        .iter()
        .map(|node| {
            let mut rs: Vec<Var> = node.inputs.iter().map(|&v| root[v]).collect();
            rs.sort_unstable();
            rs.dedup();
            rs
        })
        .collect();
    let mut unread = vec![0usize; graph.types.len()];
    for rs in &reads {
        for &r in rs {
            unread[r] += 1;
        }
    }
    let mut waiting: Vec<usize> = operands.iter().map(Vec::len).collect();
    let mut ready: Vec<usize> = (0..nodes.len()).filter(|&i| waiting[i] == 0).collect();
    let mut order = Vec::with_capacity(nodes.len());
    while !ready.is_empty() {
        let priority = |i: usize| -> (i64, usize) {
            if free(graph, i) {
                return (i64::MAX, usize::MAX);
            }
            let freed: i64 = reads[i]
                .iter()
                .filter(|&&r| !pinned[r] && unread[r] == 1)
                .map(|&r| bytes(graph, r))
                .sum();
            (freed - bytes(graph, nodes[i].output), users[i].len())
        };
        let k = (0..ready.len())
            .max_by_key(|&k| (priority(ready[k]), std::cmp::Reverse(ready[k])))
            .expect("a ready node");
        let i = ready.swap_remove(k);
        order.push(i);
        for &r in &reads[i] {
            unread[r] -= 1;
        }
        for &u in &users[i] {
            waiting[u] -= 1;
            if waiting[u] == 0 {
                ready.push(u);
            }
        }
    }
    order
}

/// Depth-first post-order schedule from the outputs, visiting each node's heavier operand
/// subgraphs first so smaller ones are computed close to their use; returns node indices.
pub(crate) fn dfs(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    let Deps { operands, users } = deps(graph);
    let total = nodes.len() as i64;
    let cap: i64 = (0..nodes.len())
        .map(|i| bytes(graph, nodes[i].output))
        .sum();
    // Per node, a saturating estimate of its subgraph's extra users and output bytes, the visit order key.
    let mut stats = vec![(0i64, 0i64); nodes.len()];
    for i in 0..nodes.len() {
        let mut s = (
            users[i].len().saturating_sub(1) as i64,
            bytes(graph, nodes[i].output),
        );
        for &o in &operands[i] {
            s = (s.0 + stats[o].0, s.1 + stats[o].1);
        }
        stats[i] = (s.0.min(total), s.1.min(cap));
    }
    let mut producer = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut visited = vec![false; nodes.len()];
    let mut order = Vec::with_capacity(nodes.len());
    let roots = graph
        .outputs()
        .iter()
        .filter_map(|&v| producer[v])
        .chain(0..nodes.len());
    for start in roots {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let sorted = |i: usize| {
            let mut ops = operands[i].clone();
            ops.sort_by_key(|&o| (stats[o].0, stats[o].1, std::cmp::Reverse(o)));
            ops
        };
        // Iterative DFS: operands are popped from the end, so the largest-stat operand is visited first.
        let mut stack = vec![(start, sorted(start))];
        while let Some((i, ops)) = stack.last_mut() {
            match ops.pop() {
                Some(o) if !visited[o] => {
                    visited[o] = true;
                    let next = sorted(o);
                    stack.push((o, next));
                }
                Some(_) => {}
                None => {
                    order.push(*i);
                    stack.pop();
                }
            }
        }
    }
    order
}

/// Overlap schedule: a topological reorder of `graph` that runs each collective, and the nodes it depends on,
/// as early as possible, and each `device_wait`, and the nodes depending on it, as late as possible, so
/// communication overlaps the compute independent of it; returns node indices, the identity when the graph
/// has no collective.
pub(crate) fn overlap(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    if !nodes.iter().any(|node| node.primitive.is_collective()) {
        return (0..nodes.len()).collect();
    }
    let Deps { operands, users } = deps(graph);
    let wait = |i: usize| matches!(nodes[i].primitive, Primitive::DeviceWait);
    // Per node, whether a collective is it or depends on it.
    let mut feeds: Vec<bool> = nodes
        .iter()
        .map(|node| node.primitive.is_collective())
        .collect();
    for i in (0..nodes.len()).rev() {
        if feeds[i] {
            operands[i].iter().for_each(|&o| feeds[o] = true);
        }
    }
    // Per node, whether it is a `device_wait` or depends on one.
    let mut waits = vec![false; nodes.len()];
    for i in 0..nodes.len() {
        waits[i] = wait(i) || operands[i].iter().any(|&o| waits[o]);
    }
    // Forward: among ready nodes, those feeding a collective first and those depending on a wait last, each
    // kind in its current order.
    let rank = |i: usize| Reverse((!feeds[i], waits[i], i));
    let mut waiting: Vec<usize> = operands.iter().map(Vec::len).collect();
    let mut ready: BinaryHeap<Reverse<(bool, bool, usize)>> = (0..nodes.len())
        .filter(|&i| waiting[i] == 0)
        .map(rank)
        .collect();
    let mut position = vec![0; nodes.len()];
    let mut t = 0;
    while let Some(Reverse((_, _, i))) = ready.pop() {
        position[i] = t;
        t += 1;
        for &u in &users[i] {
            waiting[u] -= 1;
            if waiting[u] == 0 {
                ready.push(rank(u));
            }
        }
    }
    // Backward: waits last among nodes whose users are placed, the rest in the forward order.
    let mut waiting: Vec<usize> = users.iter().map(Vec::len).collect();
    let mut ready: BinaryHeap<(bool, usize, usize)> = (0..nodes.len())
        .filter(|&i| waiting[i] == 0)
        .map(|i| (wait(i), position[i], i))
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some((_, _, i)) = ready.pop() {
        order.push(i);
        for &o in &operands[i] {
            waiting[o] -= 1;
            if waiting[o] == 0 {
                ready.push((wait(o), position[o], o));
            }
        }
    }
    order.reverse();
    order
}

impl Graph {
    /// This graph with its nodes in `order` (indices into the current nodes); types and I/O unchanged.
    pub(crate) fn reordered(&self, order: &[usize]) -> Graph {
        Graph {
            types: self.types.clone(),
            inputs: self.inputs.clone(),
            nodes: order.iter().map(|&i| self.nodes[i].clone()).collect(),
            outputs: self.outputs.clone(),
            scope: self.scope,
            sources: self.sources,
        }
    }
}
