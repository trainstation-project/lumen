//! Orders of a graph's nodes for the planner to run its steps in (XLA's
//! `hlo_memory_scheduler.cc`): each a topological order, chosen to keep
//! fewer bytes alive at once than the trace's order may. [`Plan`] plans
//! each and keeps the one whose workspace is smallest (XLA's
//! `DefaultMemoryScheduler`, which keeps the one whose estimate is).
//!
//! [`Plan`]: super::Plan

use super::{Graph, Primitive, Var};

/// Which node defines each value, and the nodes each value is read by
/// (each once). A custom op is also read by the next one, so that they
/// stay in the trace's order: their kernels may have effects the graph
/// does not show.
struct Deps {
    operands: Vec<Vec<usize>>,
    users: Vec<Vec<usize>>,
}

fn deps(graph: &Graph) -> Deps {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut operands: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut last_custom = None;
    for (i, node) in nodes.iter().enumerate() {
        let mut ops: Vec<usize> = node.inputs.iter().filter_map(|&v| producer[v]).collect();
        if let Primitive::CustomCall { .. } = node.primitive {
            ops.extend(last_custom);
            last_custom = Some(i);
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

/// Each value's bytes: a reshape's are its operand's (the same memory, in
/// the plan), so its own count nothing.
fn roots(graph: &Graph) -> Vec<Var> {
    let mut root: Vec<Var> = (0..graph.types.len()).collect();
    for node in graph.nodes() {
        if let Primitive::Reshape { .. } = node.primitive {
            root[node.output] = root[node.inputs[0]];
        }
    }
    root
}

fn bytes(graph: &Graph, v: Var) -> i64 {
    let ty = graph.type_of(v);
    (ty.numel() * ty.dtype.size_of()) as i64
}

/// Whether node `i` takes no memory of its own worth ordering by: a
/// reshape (its operand's memory), a fusion's other output (written by the
/// fusion), or a scalar. Scheduled as soon as it can be (XLA clusters
/// scalars).
fn free(graph: &Graph, i: usize) -> bool {
    let node = &graph.nodes()[i];
    matches!(
        node.primitive,
        Primitive::Reshape { .. } | Primitive::FusionOutput { .. }
    ) || graph.type_of(node.output).numel() <= 1
}

/// XLA's `ListScheduler`: of the nodes whose operands have run, the one
/// that frees the most bytes less those it defines, then the one with the
/// most users (whose value more nodes wait on), then the trace's first.
/// The graph's inputs and outputs are never freed.
pub(crate) fn list(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    let Deps { operands, users } = deps(graph);
    let root = roots(graph);
    let mut pinned = vec![false; graph.types.len()];
    for &v in graph.inputs().iter().chain(graph.outputs()) {
        pinned[root[v]] = true;
    }
    // The nodes still to read each root's memory.
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
        // The highest priority; of equal ones, the trace's first.
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

/// XLA's `DFSMemoryScheduler`: a post-order from the outputs, each node's
/// operands visited in decreasing order of the users beyond the first of
/// the nodes they depend on (a value many read is computed early), then of
/// those nodes' bytes, then the trace's order. Nodes no output depends on
/// follow, in the trace's order.
pub(crate) fn dfs(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    let Deps { operands, users } = deps(graph);
    let total = nodes.len() as i64;
    let cap: i64 = (0..nodes.len())
        .map(|i| bytes(graph, nodes[i].output))
        .sum();
    // (extra users, bytes), each of the node and every node it depends on
    // (counted once a path, capped as XLA's are).
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
        // (node, its operands still to visit, by priority, last first).
        visited[start] = true;
        let sorted = |i: usize| {
            let mut ops = operands[i].clone();
            ops.sort_by_key(|&o| (stats[o].0, stats[o].1, std::cmp::Reverse(o)));
            ops
        };
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

impl Graph {
    /// The graph with its nodes in `order` (a topological one), each value
    /// numbered as it is.
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
