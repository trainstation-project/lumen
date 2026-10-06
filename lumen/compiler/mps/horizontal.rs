//! Horizontal loop fusion (XLA's HorizontalLoopFusion, see
//! xla/service/gpu/transforms/horizontal_loop_fusion.cc before XLA removed
//! it in 50f16da8fb, and horizontal_input_fusion.cc before 7c1802e121):
//! independent loop fusions, none reading another's value, each too small
//! for its launch's cost (an optimizer's update of each parameter, a cast
//! of each weight), are one kernel: a multi-output loop fusion computing
//! each of them at every index (XLA's case 1, of same-shaped fusions), its
//! outputs their values, each in its own buffer.
//!
//! XLA concatenated fusions of different shapes into one buffer (its case
//! 2), then sliced it: copies, and the memory XLA removed the pass over
//! ("It does not add anything to the benchmarks except for OOMs"). Here
//! only fusions of as many elements are one kernel, so a thread computes
//! every member at its index: each member reads and writes its own buffers
//! at that index, as it did alone (an output written in place over an input
//! stays in one thread), and computes exactly what it did alone.
//!
//! XLA fused the operands of one consumer (the ROOT tuple), each read by it
//! alone, so no fusion could make a cycle. Here a fusion joins a group of
//! as many elements unless that makes one (each group one node, nothing
//! may reach a group from the group): an optimizer's moments' updates one
//! kernel, its parameters' updates, reading them, another. A group runs
//! where its last member did, the steps reading its values before it
//! after it: members of groups alone (another group's, run later), so no
//! other step moves. The members' operands live until it runs: those that
//! would have died sooner may hold at most [`MAX_EXTENDED_BYTES`].

use std::collections::HashMap;

use super::{codegen, fusion, split_k};
use crate::compiler::attention;
use crate::graph::{Graph, Node, Primitive, Var, intern};

/// The most fusions in one kernel (XLA's `kMaxFusionBatchSize`).
const MAX_MEMBERS: usize = 32;

/// The buffers a kernel binds (31), less its element count.
const MAX_BUFFERS: usize = 30;

/// The most elements a member computes: above it, its launch costs little
/// beside its work (XLA's `kShapeThreshold`, smaller: a Metal launch's
/// cost is a few microseconds of a kernel of this many elements).
const MAX_ELEMENTS: usize = 1 << 20;

/// The groups of each size still growing: a fusion joins the first it can
/// (an optimizer's moments' updates one, its parameters' another).
const MAX_OPEN: usize = 4;

/// The most bytes of workspace values a kernel may keep alive longer than
/// its members alone would.
const MAX_EXTENDED_BYTES: usize = 16 << 20;

/// A group of fusions, one kernel: its members (in order), the values they
/// read, how many values they write, and the step it runs at (once every
/// value it reads is computed, not before its first member).
struct Group {
    members: Vec<usize>,
    reads: Vec<Var>,
    values: usize,
    at: usize,
}

/// `graph` (loop-fused, [`fusion::fuse`]) with its independent loop
/// fusions of as many elements, and its lone elementwise primitives,
/// fused horizontally: each group one [`Primitive::Fusion`], its kernel
/// named by `kernel` (given its body and which of its inputs it takes by
/// value: `scalars`).
pub(crate) fn fuse(
    graph: &Graph,
    scalars: &[Var],
    mut kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    // Each node's values: its own, then its fusion outputs' (by index).
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
    // Each value's readers (a fusion output reads its fusion: not one).
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| !output_of[i]) {
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }
    let last_read = |v: Var| readers[v].iter().copied().max();
    let successors = |i: usize| values[i].iter().flat_map(|&v| readers[v].iter().copied());
    // Whether the nodes `members`, one kernel, would reach it: from what
    // reads their values, forward, each group's members one node.
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

    // The step computing each value (a fusion output's, its fusion).
    let step_of = |v: Var| {
        producer[v].map(|p| match output_of[p] {
            true => producer[nodes[p].inputs[0]].expect("a fusion's output"),
            false => p,
        })
    };
    // Whether the nodes `members`, one kernel run at `at`, would move a
    // node of no group later: one reading their values before `at`, or
    // reading a group's that it moves so.
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
    // Groups still growing, by their members' elements and (tiled as a
    // transpose: by their tile, so of one shape) their shape.
    let mut open: HashMap<(usize, Option<Vec<usize>>), Vec<usize>> = HashMap::new();
    for (c, node) in nodes.iter().enumerate() {
        // Each of a fusion's outputs read as a value of the graph.
        let whole = match &node.primitive {
            Primitive::Fusion { body, .. } => body.outputs().len() == values[c].len(),
            _ => true,
        };
        if !candidate(graph, node) || !whole {
            continue;
        }
        let ty = graph.type_of(node.output);
        let tiled = match &node.primitive {
            Primitive::Fusion { body, .. } => codegen::transpose_tiling(body).is_some(),
            _ => false,
        };
        let key = (ty.numel(), tiled.then(|| ty.shape.clone()));
        let mut mine = node.inputs.clone();
        mine.sort_unstable();
        mine.dedup();
        let count = values[c].len();
        let joins = |g: &Group| {
            let mut all = g.reads.clone();
            all.extend(&mine);
            all.sort_unstable();
            all.dedup();
            let mut members = g.members.clone();
            members.push(c);
            // Where it would run: once its reads are (a group's, once it
            // runs), not before its first member.
            let at = all
                .iter()
                .filter_map(|&v| step_of(v))
                .map(|p| group_of[p].map_or(p, |h| groups[h].at) + 1)
                .fold(members[0], usize::max);
            // Values kept alive longer: those its members before `at` read
            // that would die before it, those after it write, written
            // sooner.
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
            // A tile holds each transpose read through it.
            let transposes = |m: &usize| match &nodes[*m].primitive {
                Primitive::Fusion { body, .. } => body
                    .nodes()
                    .iter()
                    .filter(|n| matches!(n.primitive, Primitive::Transpose { .. }))
                    .count(),
                _ => 0,
            };
            let heroes =
                !tiled || members.iter().map(transposes).sum::<usize>() <= codegen::MAX_HEROES;
            let fits = g.members.len() < MAX_MEMBERS
                && heroes
                && all.len() + g.values + count <= MAX_BUFFERS
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
                // The oldest stops growing.
                if growing.len() > MAX_OPEN {
                    growing.remove(0);
                }
            }
        }
    }

    // Each group of more than one member: its body and values, tiled as a
    // transpose if (and only if) its members were (of one shape, each
    // computed at the tile's index). Its members are otherwise left as
    // they are (a node alone makes no cycle).
    let mut bodies: Vec<Option<(Graph, Vec<Var>)>> = Vec::new();
    for g in &groups {
        let tiled = |m: &usize| match &nodes[*m].primitive {
            Primitive::Fusion { body, .. } => codegen::transpose_tiling(body).is_some(),
            _ => false,
        };
        let made = (g.members.len() > 1)
            .then(|| body(graph, &g.members, &g.reads, &values))
            .filter(|(body, _)| codegen::transpose_tiling(body).is_some() == tiled(&g.members[0]));
        if made.is_none() {
            g.members.iter().for_each(|&m| group_of[m] = None);
        }
        bodies.push(made);
    }
    let member_output = |i: usize| {
        output_of[i] && producer[nodes[i].inputs[0]].is_some_and(|p| group_of[p].is_some())
    };

    // The order of the steps: the graph's, each group's once the values it
    // reads are (from its first member's on), a step reading a value of a
    // group not yet run after it.
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
                let made = match group_of[w] {
                    Some(k) => groups[k].members.clone(),
                    None => vec![w],
                };
                for m in made {
                    values[m].iter().for_each(|&v| ready[v] = true);
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
            out.set_scope(node.scope);
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a graph is typed as the original");
            continue;
        };
        let g = &groups[k];
        let (body, outputs) = bodies[k].take().expect("a group's body");
        let by_value: Vec<bool> = g.reads.iter().map(|v| scalars.contains(v)).collect();
        let name = kernel(&body, &by_value);
        // Its members' labels, as merged dots' (`Nx dot_general`): each
        // label once, in order, `Nx` before one N members share.
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
        // The ranges its members share (calls of the same outer ranges).
        let first = nodes[g.members[0]].scope;
        let shared = g.members.iter().fold(first.len(), |n, &m| {
            first
                .iter()
                .zip(nodes[m].scope)
                .take(n)
                .take_while(|(a, b)| a == b)
                .count()
        });
        out.set_scope(&first[..shared]);
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

/// Whether `node` may be a member: a loop fusion (perhaps tiled as a
/// transpose; no contraction, reduction or attention: each its own kernel)
/// or a lone elementwise primitive, of at most [`MAX_ELEMENTS`] elements.
fn candidate(graph: &Graph, node: &Node) -> bool {
    let elements = graph.type_of(node.output).numel();
    if elements == 0 || elements > MAX_ELEMENTS {
        return false;
    }
    match &node.primitive {
        Primitive::Fusion { body, .. } => {
            let reduces = body.nodes().iter().any(|n| {
                matches!(
                    n.primitive,
                    Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                )
            });
            !reduces
                && attention::of_body(body).is_none()
                && attention::backward_of_body(body).is_none()
                && split_k::atomic_dot(body).is_none()
                && codegen::gemm_dot(body).is_none()
        }
        p => fusion::elementwise(p) && fusion::fusible(graph, node),
    }
}

/// A member's label: a fusion's, or its primitive's.
fn label(graph: &Graph, node: &Node) -> &'static str {
    match &node.primitive {
        Primitive::Fusion { label, .. } => label,
        _ => graph.label(node),
    }
}

/// The body of a group of `members` reading `reads`: each member's
/// computation (a fusion's body, or its primitive) on its inputs, its
/// outputs every member's `values` in order; and those values.
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
                    let v = body
                        .apply(n.primitive.clone(), &ins)
                        .expect("a member's body is typed");
                    local.insert(n.output, v);
                }
                written.extend(inner.outputs().iter().map(|v| local[v]));
            }
            p => written.push(body.apply(p.clone(), &inputs).expect("a member is typed")),
        }
        outputs.extend(&values[m]);
    }
    body.set_outputs(&written)
        .expect("the members' values are values of the body");
    (body, outputs)
}
