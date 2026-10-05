//! Rematerialization (XLA's `hlo_rematerialization.cc`): while a plan's
//! workspace exceeds a memory limit, a value is computed again where it is
//! read later, rather than kept alive until then (a forward activation the
//! backward reads, recomputed in the backward). As XLA's, each pick is the
//! value whose recomputation saves the most memory, whatever it costs to
//! compute: running out of memory is worse than running slower. Here each
//! candidate's saving is measured by planning it.

use super::{Graph, Node, Plan, Primitive, Var};

/// Whether node `node` may be computed again: not a custom op (whose kernel
/// may have effects the graph does not show, XLA's `kCustomCall`), a fusion
/// with other outputs (written by its one step), one of those, nor a
/// reshape (its operand's memory: nothing to save).
fn recomputable(graph: &Graph, node: &Node) -> bool {
    let multi_output = graph.nodes().iter().any(|n| {
        matches!(n.primitive, Primitive::FusionOutput { .. }) && n.inputs[0] == node.output
    });
    !multi_output
        && !matches!(
            node.primitive,
            Primitive::CustomCall { .. } | Primitive::FusionOutput { .. } | Primitive::Reshape { .. }
        )
}

/// `graph` with node `i`'s value computed again right before its `k`th
/// reader (in the nodes' order): that reader and the later ones read the
/// copy. With `k` = 0, node `i` moves there (it is no longer read where it
/// was; the planner drops it). With `depth` > 0, the nodes computing its
/// operands are computed again with it, that many levels back (XLA's block
/// rematerialization): those operands need not be kept alive until then
/// either (a slice of a dot's output, the dot recomputed with it).
fn recompute(graph: &Graph, i: usize, k: usize, depth: usize) -> Graph {
    let v = graph.nodes[i].output;
    let at = (i + 1..graph.nodes.len())
        .filter(|&j| graph.nodes[j].inputs.contains(&v))
        .nth(k)
        .expect("a kth reader");
    // The block: node `i` and, `depth` levels back, the recomputable nodes
    // computing its operands; in the nodes' order.
    let mut block = vec![i];
    let mut level = vec![i];
    for _ in 0..depth {
        let mut next = Vec::new();
        for &j in &level {
            for &u in &graph.nodes[j].inputs {
                if let Some(p) = graph.nodes[..j].iter().position(|n| n.output == u)
                    && recomputable(graph, &graph.nodes[p])
                    && !block.contains(&p)
                {
                    block.push(p);
                    next.push(p);
                }
            }
        }
        level = next;
    }
    block.sort_unstable();
    let mut g = graph.clone();
    let mut copies = Vec::new();
    for (n, &j) in block.iter().enumerate() {
        let node = &graph.nodes[j];
        let copy = g.types.len();
        g.types.push(g.types[node.output].clone());
        let inputs = node
            .inputs
            .iter()
            .map(|u| copies.iter().find(|&&(o, _)| o == *u).map_or(*u, |&(_, c)| c))
            .collect();
        g.nodes.insert(
            at + n,
            Node {
                primitive: node.primitive.clone(),
                inputs,
                output: copy,
            },
        );
        copies.push((node.output, copy));
    }
    let copy = copies.iter().find(|&&(o, _)| o == v).expect("the value's copy").1;
    for later in &mut g.nodes[at + block.len()..] {
        for input in &mut later.inputs {
            if *input == v {
                *input = copy;
            }
        }
    }
    g
}

/// The bytes alive at each node of `graph`, in order (XLA's
/// `MemoryUsageTracker`, without the planner's in-place reuse and
/// fragmentation): each value, a reshape's its operand's, from the node
/// computing it to its last reader; an output's to the end.
fn usage(graph: &Graph) -> Vec<usize> {
    let n = graph.nodes.len();
    let mut root: Vec<Var> = (0..graph.types.len()).collect();
    let mut first = vec![usize::MAX; graph.types.len()];
    let mut last = vec![0; graph.types.len()];
    for (t, node) in graph.nodes.iter().enumerate() {
        if let Primitive::Reshape { .. } = node.primitive {
            root[node.output] = root[node.inputs[0]];
        }
        // A fusion's other output is written by the fusion.
        let r = match node.primitive {
            Primitive::FusionOutput { .. } => node.output,
            _ => root[node.output],
        };
        first[r] = first[r].min(t);
        for &u in &node.inputs {
            last[root[u]] = t;
        }
    }
    for &o in &graph.outputs {
        last[root[o]] = n;
    }
    // Each value's bytes added where it starts and taken away after it ends.
    let mut delta = vec![0i64; n + 2];
    for r in (0..graph.types.len()).filter(|&r| root[r] == r && first[r] != usize::MAX) {
        // A value nothing reads is dead: the planner drops its node.
        if last[r] < first[r] {
            continue;
        }
        let ty = graph.type_of(r);
        let bytes = (ty.numel() * ty.dtype.size_of()) as i64;
        delta[first[r]] += bytes;
        delta[last[r] + 1] -= bytes;
    }
    let mut live = 0;
    delta[..n]
        .iter()
        .map(|d| {
            live += d;
            live as usize
        })
        .collect()
}

/// `graph` (its nodes in the order to run them) rematerialized until
/// `plan` of it needs at most `limit` bytes of workspace, or no
/// recomputation saves any: the smallest plan measured. As XLA's, candidates are ranked
/// by a model of the bytes alive at each node ([`usage`]): each a value
/// alive across the node where most are (the peak), computed again (with
/// its operands, [`recompute`]) at its first reader after it; the one
/// lowering the peak most is taken, until the model's peak is under a
/// target. Then the plan is measured, and the target lowered by what the
/// plan needs beyond the model, a few times. Values smaller than
/// `MIN_REMAT_BYTES` are not recomputed (XLA's `min_remat_size`).
pub(crate) fn rematerialize(graph: &Graph, limit: usize, plan: impl Fn(&Graph) -> Plan) -> Plan {
    const MIN_REMAT_BYTES: usize = 64 << 10;
    // The levels of operands recomputed with a value, at most.
    const MAX_DEPTH: usize = 2;
    // The plans measured, at most.
    const ROUNDS: usize = 4;
    let mut graph = graph.clone();
    let mut best = plan(&graph);
    let mut target = limit;
    // At most a recomputation a node of the graph, in all.
    let mut budget = graph.nodes.len();
    for _ in 0..ROUNDS {
        if best.workspace_bytes() <= limit {
            break;
        }
        let before = graph.nodes.len();
        while budget > 0 {
            let live = usage(&graph);
            let (peak_at, &peak) = live
                .iter()
                .enumerate()
                .max_by_key(|&(t, &b)| (b, std::cmp::Reverse(t)))
                .expect("a node");
            if peak <= target {
                break;
            }
            let mut pick: Option<((usize, usize), Graph)> = None;
            for (i, node) in graph.nodes[..peak_at].iter().enumerate() {
                let ty = graph.type_of(node.output);
                if ty.numel() * ty.dtype.size_of() < MIN_REMAT_BYTES
                    || graph.outputs.contains(&node.output)
                    || !recomputable(&graph, node)
                {
                    continue;
                }
                let readers: Vec<usize> = (i + 1..graph.nodes.len())
                    .filter(|&j| graph.nodes[j].inputs.contains(&node.output))
                    .collect();
                // Alive across the peak: read after it, not at it.
                if readers.contains(&peak_at) || readers.last().is_none_or(|&j| j < peak_at) {
                    continue;
                }
                let k = readers.iter().filter(|&&j| j < peak_at).count();
                for depth in 0..=MAX_DEPTH {
                    let candidate = recompute(&graph, i, k, depth);
                    let live = usage(&candidate);
                    // The new peak, then what is alive where the old one was.
                    let score = (*live.iter().max().expect("a node"), live[peak_at]);
                    if pick.as_ref().is_none_or(|(s, _)| score < *s) {
                        pick = Some((score, candidate));
                    }
                }
            }
            // Taken if it lowers the peak, or frees a value's worth there.
            match pick {
                Some(((p, at), candidate)) if p < peak || at + MIN_REMAT_BYTES <= peak => {
                    graph = candidate;
                    budget -= 1;
                }
                _ => break,
            }
        }
        if graph.nodes.len() == before {
            break;
        }
        let measured = plan(&graph);
        if measured.workspace_bytes() >= best.workspace_bytes() {
            break;
        }
        // The plan's bytes beyond the model's (inputs copied in, kernel
        // scratch, fragmentation), taken off the target.
        let model = usage(&graph).into_iter().max().unwrap_or(0);
        target = limit.saturating_sub(measured.workspace_bytes().saturating_sub(model));
        best = measured;
    }
    best
}
