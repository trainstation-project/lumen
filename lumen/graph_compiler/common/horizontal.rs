//! Horizontal fusion: merges independent kernels of the same kind into one launch.

use std::collections::HashMap;

use crate::graph::{Graph, Node, Primitive, Scope, Var, intern};

/// Most nodes in one horizontal group.
const MAX_MEMBERS: usize = 32;

/// Groups per key still accepting members; the oldest is closed when more open.
const MAX_OPEN: usize = 4;

/// Most bytes whose buffer lifetime a group may extend by moving reads later or writes earlier.
const MAX_EXTENDED_BYTES: usize = 16 << 20;

/// Nodes fused into one kernel.
struct Group {
    /// Member node indices, in graph order.
    members: Vec<usize>,
    /// Values the members read, deduplicated and sorted.
    reads: Vec<Var>,
    /// Number of values the members produce.
    values: usize,
    /// Earliest node position the group can run at, after all its operands exist.
    at: usize,
}

/// A backend's rules for horizontal fusion.
pub(crate) trait Target {
    /// Whether `node` may join a horizontal group.
    fn candidate(&self, graph: &Graph, node: &Node) -> bool;
    /// Nodes with equal keys (in the same scope) may share a group.
    fn key(&self, graph: &Graph, node: &Node) -> Vec<usize>;
    /// Whether `members` fit one kernel taking `buffers` buffers.
    fn fits(&self, graph: &Graph, members: &[usize], buffers: usize) -> bool;
    /// Final check of a group's fused body.
    fn accepts(&self, graph: &Graph, members: &[usize], body: &Graph) -> bool;
    /// Requests a kernel for `body`, with inputs flagged in `by_value` passed by value; returns its name.
    fn kernel(&mut self, body: &Graph, by_value: &[bool]) -> String;
}

/// Groups independent candidate nodes with equal keys into fusions, keeping the graph acyclic and
/// bounding how much buffer lifetime grows. `scalars` are values passed to kernels by value.
pub(crate) fn fuse(graph: &Graph, scalars: &[Var], target: &mut impl Target) -> Graph {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    // Values each node produces, including the extra outputs of a fusion (its `FusionOutput` readers).
    let mut values: Vec<Vec<Var>> = nodes.iter().map(|node| vec![node.output]).collect();
    let mut output_of = vec![false; nodes.len()];
    let mut extra: Vec<Vec<(usize, Var)>> = vec![Vec::new(); nodes.len()];

    for (f, node) in nodes.iter().enumerate() {
        if let Primitive::FusionOutput { index, .. } = node.primitive {
            let p = producer[node.inputs[0]].expect("a fusion's output");
            extra[p].push((index, node.output));
            output_of[f] = true;
        }
    }

    for (i, e) in extra.iter_mut().enumerate() {
        e.sort_unstable();
        values[i].extend(e.iter().map(|&(_, v)| v));
    }

    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| !output_of[i]) {
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }

    let last_read = |v: Var| readers[v].iter().copied().max();
    let successors = |i: usize| values[i].iter().flat_map(|&v| readers[v].iter().copied());

    // Whether some path from the members' readers leads back into the members.
    let cycles = |members: &[usize], group_of: &[Option<usize>], groups: &[Group]| {
        let (mut seen, mut stack) = (vec![false; nodes.len()], Vec::new());
        stack.extend(members.iter().flat_map(|&m| successors(m)));

        while let Some(y) = stack.pop() {
            if std::mem::replace(&mut seen[y], true) {
                continue;
            }

            if members.contains(&y) {
                return true;
            }

            match group_of[y] {
                Some(h) => stack.extend(groups[h].members.iter().flat_map(|&m| successors(m))),
                None => stack.extend(successors(y)),
            }
        }

        false
    };

    let step_of = |v: Var| {
        producer[v].map(|p| match output_of[p] {
            true => producer[nodes[p].inputs[0]].expect("a fusion's output"),
            false => p,
        })
    };

    // Whether a node outside any group, before `at`, depends on a member and would have to wait.
    let defers = |members: &[usize], at: usize, group_of: &[Option<usize>], groups: &[Group]| {
        let before = |m: &usize| successors(*m).filter(|&r| r < at).collect::<Vec<_>>();
        let mut stack: Vec<usize> = members.iter().flat_map(before).collect();
        let mut seen = vec![false; nodes.len()];

        while let Some(y) = stack.pop() {
            if members.contains(&y) || std::mem::replace(&mut seen[y], true) {
                continue;
            }

            match group_of[y] {
                Some(h) => stack.extend(groups[h].members.iter().flat_map(before)),
                None => return true,
            }
        }

        false
    };

    let mut groups: Vec<Group> = Vec::new();
    let mut group_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let bytes = |v: Var| graph.type_of(v).numel() * graph.type_of(v).dtype.size_of();
    type Key = (Vec<usize>, Scope);
    let mut open: HashMap<Key, Vec<usize>> = HashMap::new();

    for (c, node) in nodes.iter().enumerate() {
        let whole = match &node.primitive {
            Primitive::Fusion { body, .. } => body.outputs().len() == values[c].len(),
            _ => true,
        };

        if !target.candidate(graph, node) || !whole {
            continue;
        }

        let key = (target.key(graph, node), node.scope);
        let mut mine = node.inputs.clone();
        mine.sort_unstable();
        mine.dedup();
        let count = values[c].len();

        // Joining `g` runs the group at `at`: reads before it live longer, members after it write earlier.
        let joins = |g: &Group| {
            let mut all = g.reads.clone();
            all.extend(&mine);
            all.sort_unstable();
            all.dedup();
            let mut members = g.members.clone();
            members.push(c);

            let at = all
                .iter()
                .filter_map(|&v| step_of(v))
                .map(|p| group_of[p].map_or(p, |h| groups[h].at) + 1)
                .fold(members[0], usize::max);

            let read: usize = all
                .iter()
                .filter(|&&v| producer[v].is_some() && last_read(v).is_some_and(|r| r < at))
                .map(|&v| bytes(v))
                .sum();

            let written: usize = members
                .iter()
                .filter(|&&m| m > at)
                .flat_map(|&m| values[m].iter().map(|&v| bytes(v)))
                .sum();

            let fits = g.members.len() < MAX_MEMBERS
                && target.fits(graph, &members, all.len() + g.values + count)
                && read + written <= MAX_EXTENDED_BYTES
                && !cycles(&members, &group_of, &groups)
                && !defers(&members, at, &group_of, &groups);

            fits.then_some(at)
        };

        let growing = open.entry(key).or_default();

        let found = growing
            .iter()
            .find_map(|&k| joins(&groups[k]).map(|at| (k, at)));

        match found {
            Some((k, at)) => {
                let g = &mut groups[k];
                g.members.push(c);
                g.reads.extend(&mine);
                g.reads.sort_unstable();
                g.reads.dedup();
                g.values += count;
                g.at = at;
                group_of[c] = Some(k);
            }
            None => {
                group_of[c] = Some(groups.len());
                growing.push(groups.len());

                groups.push(Group {
                    members: vec![c],
                    reads: mine,
                    values: count,
                    at: c,
                });

                if growing.len() > MAX_OPEN {
                    growing.remove(0);
                }
            }
        }
    }

    let mut bodies: Vec<Option<(Graph, Vec<Var>)>> = Vec::new();

    for g in &groups {
        let made = (g.members.len() > 1)
            .then(|| body(graph, &g.members, &g.reads, &values))
            .filter(|(body, _)| target.accepts(graph, &g.members, body));

        if made.is_none() {
            g.members.iter().for_each(|&m| group_of[m] = None);
        }

        bodies.push(made);
    }

    let member_output = |i: usize| {
        output_of[i] && producer[nodes[i].inputs[0]].is_some_and(|p| group_of[p].is_some())
    };

    // Emit nodes in an order where each group runs once all its reads are ready.
    let order = {
        let mut ready = vec![false; n];
        graph.inputs().iter().for_each(|&v| ready[v] = true);
        let (mut order, mut waiting) = (Vec::new(), Vec::new());

        for i in 0..nodes.len() {
            let step = match group_of[i] {
                Some(k) => groups[k].members[0] == i,
                None => !member_output(i),
            };

            if step {
                waiting.push(i);
            }

            while let Some(w) = waiting.iter().position(|&w| match group_of[w] {
                Some(k) => groups[k].reads.iter().all(|&v| ready[v]),
                None => nodes[w].inputs.iter().all(|&v| ready[v]),
            }) {
                let w = waiting.remove(w);

                match group_of[w] {
                    Some(k) => {
                        for &m in &groups[k].members {
                            values[m].iter().for_each(|&v| ready[v] = true);
                        }
                    }
                    None => ready[nodes[w].output] = true,
                }

                order.push(w);
            }
        }

        assert!(
            waiting.is_empty(),
            "horizontal fusion keeps the graph acyclic"
        );

        order
    };

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; n];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for i in order {
        let node = &nodes[i];

        let Some(k) = group_of[i] else {
            out.set_origin(node);
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a graph is typed as the original");

            out.set_label(map[node.output], node.label);
            continue;
        };

        let g = &groups[k];
        let (body, outputs) = bodies[k].take().expect("a group's body");
        let by_value: Vec<bool> = g.reads.iter().map(|v| scalars.contains(v)).collect();
        let name = target.kernel(&body, &by_value);
        let mut labels: Vec<(&str, usize)> = Vec::new();

        for &m in &g.members {
            let l = label(graph, &nodes[m]);

            match labels.iter_mut().find(|(seen, _)| *seen == l) {
                Some((_, count)) => *count += 1,
                None => labels.push((l, 1)),
            }
        }

        let labels: Vec<String> = labels
            .into_iter()
            .map(|(l, count)| match count {
                1 => l.to_owned(),
                n => format!("{n}x {l}"),
            })
            .collect();

        let label = intern(labels.join(" | "));
        out.set_origin(&nodes[g.members[0]]);
        let reads: Vec<Var> = g.reads.iter().map(|&v| map[v]).collect();

        let fused = out
            .apply(Primitive::Fusion { name, label, body }, &reads)
            .expect("a fusion of its members' values");

        map[outputs[0]] = fused;

        for (index, &v) in outputs.iter().enumerate().skip(1) {
            let ty = graph.type_of(v).clone();

            map[v] = out
                .apply(Primitive::FusionOutput { index, ty }, &[fused])
                .expect("the fusion's output");
        }
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();

    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");

    out
}

/// A fused body's outputs grouped by element count, in order of first appearance: each group's
/// element count and output positions. A kernel runs each group on its own range of the grid, so
/// a horizontal fusion of several element counts is one launch.
pub(crate) fn segments(body: &Graph) -> Vec<(usize, Vec<usize>)> {
    let mut segments: Vec<(usize, Vec<usize>)> = Vec::new();

    for (k, &v) in body.outputs().iter().enumerate() {
        let numel = body.type_of(v).numel();

        match segments.iter_mut().find(|(n, _)| *n == numel) {
            Some((_, outputs)) => outputs.push(k),
            None => segments.push((numel, vec![k])),
        }
    }

    segments
}

/// Display label of a node, using a fusion's own label.
fn label(graph: &Graph, node: &Node) -> &'static str {
    match &node.primitive {
        Primitive::Fusion { label, .. } => label,
        _ => graph.label(node),
    }
}

/// The fused body of `members` reading `reads` (inlining member fusions), and the graph values its
/// outputs correspond to.
fn body(graph: &Graph, members: &[usize], reads: &[Var], values: &[Vec<Var>]) -> (Graph, Vec<Var>) {
    let mut body = Graph::new();
    let mut var: HashMap<Var, Var> = HashMap::new();

    for &v in reads {
        var.insert(v, body.input(graph.type_of(v).clone()));
    }

    let (mut outputs, mut written) = (Vec::new(), Vec::new());

    for &m in members {
        let node = &graph.nodes()[m];
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();

        match &node.primitive {
            Primitive::Fusion { body: inner, .. } => {
                let mut local: HashMap<Var, Var> =
                    inner.inputs().iter().copied().zip(inputs).collect();

                for n in inner.nodes() {
                    let ins: Vec<Var> = n.inputs.iter().map(|v| local[v]).collect();
                    body.set_origin(n);

                    let v = body
                        .apply(n.primitive.clone(), &ins)
                        .expect("a member's body is typed");

                    local.insert(n.output, v);
                }

                written.extend(inner.outputs().iter().map(|v| local[v]));
            }
            p => {
                body.set_origin(node);
                written.push(body.apply(p.clone(), &inputs).expect("a member is typed"))
            }
        }

        outputs.extend(&values[m]);
    }

    body.set_outputs(&written)
        .expect("the members' values are values of the body");

    (body, outputs)
}
