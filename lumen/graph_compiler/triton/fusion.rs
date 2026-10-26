//! Triton loop fusion: groups elementwise chains, reductions and GEMM epilogues into Triton kernels.

use crate::codegen::regions::gemm_regions;
use crate::codegen::triton::fusible;
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::common::loops::fused;
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::graph_compiler::{constant, elementwise};

/// Compiler pass fusing nodes into Triton loop kernels.
pub(super) struct LoopFusion;

impl<M: Module> CompilerPass<M> for LoopFusion {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    /// Replaces each group with a `loop` fusion rooted at its last node.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = &recomputed_constants(m.graph());
        let (group, extra) = groups(graph);
        *m.graph_mut() = fused(
            graph,
            &group,
            |root, members| {
                let main = members.iter().find(|&&k| {
                    let p = &graph.nodes()[k].primitive;
                    reduction(p) || matches!(p, Primitive::DotGeneral { .. })
                });
                ("loop", *main.unwrap_or(&root))
            },
            |root| extra[root].clone(),
        );
        Ok(())
    }
}

/// `graph` with each fusible constant (a `full` and the layout or elementwise ops on constants
/// only) recomputed for every node reading it, so a constant several loops share joins each loop
/// instead of running as a kernel of its own.
fn recomputed_constants(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let shared = |v: Var| {
        constant(graph, &producer, v) && producer[v].is_some_and(|p| fusible(graph, &nodes[p]))
    };
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    // Emits a fresh copy of constant `v`'s chain, or returns the value already mapped.
    fn read(
        v: Var,
        graph: &Graph,
        producer: &[Option<usize>],
        shared: &dyn Fn(Var) -> bool,
        map: &[Var],
        out: &mut Graph,
    ) -> Var {
        if !shared(v) {
            return map[v];
        }
        let node = &graph.nodes()[producer[v].expect("a constant's producer")];
        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&u| read(u, graph, producer, shared, map, out))
            .collect();
        out.set_origin(node);
        let copy = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a constant typed as the original");
        out.set_label(copy, node.label);
        copy
    }
    for node in nodes {
        if shared(node.output) {
            continue;
        }
        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&u| read(u, graph, &producer, &shared, &map, &mut out))
            .collect();
        out.set_origin(node);
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node typed as the original");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph
        .outputs()
        .iter()
        .map(|&v| read(v, graph, &producer, &shared, &map, &mut out))
        .collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    out
}

/// A GEMM with an epilogue over its output column regions, claimed before ordinary grouping.
struct Claim {
    /// The dot, its region chains and constants only they read.
    members: Vec<usize>,
    /// Last node whose value leaves the group; the fusion's root.
    main: usize,
    /// Other values that leave the group, as extra fusion outputs.
    extra: Vec<Var>,
}

/// Claims each dot whose output is split into column regions, with the elementwise chains reading
/// each region, when every value leaving a region has the region's element count.
fn claims(graph: &Graph) -> Vec<Claim> {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        for &v in &node.inputs {
            users[v].push(i);
        }
    }
    let output = |v: Var| graph.outputs().contains(&v);
    let mut claimed = vec![false; nodes.len()];
    let mut found = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.primitive, Primitive::DotGeneral { .. }) || claimed[i] {
            continue;
        }
        let Some(regions) = gemm_regions(graph, node) else {
            continue;
        };
        // Per region: the slice/reshape nodes leading from the dot to each member.
        let mut sets: Vec<Vec<usize>> = regions
            .iter()
            .map(|c| {
                let mut set = Vec::new();
                for v in c.members() {
                    let mut k = producer[v].expect("a slice of the dot");
                    loop {
                        if !set.contains(&k) {
                            set.push(k);
                        }
                        match producer[nodes[k].inputs[0]] {
                            Some(p) if p != i => k = p,
                            _ => break,
                        }
                    }
                }
                set
            })
            .collect();
        let mut region_of: Vec<Option<usize>> = vec![None; n];
        for (r, set) in sets.iter().enumerate() {
            for &k in set {
                region_of[nodes[k].output] = Some(r);
            }
        }
        let shape = &graph.type_of(node.output).shape;
        // The nodes each region set exposes, and the last of them, when that is a valid fusion root.
        let ends = |sets: &[Vec<usize>]| -> Option<(Vec<usize>, usize)> {
            let mut ends = Vec::new();
            for (c, set) in regions.iter().zip(sets) {
                let size = graph.type_of(node.output).numel() / shape[c.axis] * c.size();
                let before = ends.len();
                for &k in set {
                    let v = nodes[k].output;
                    if output(v) || users[v].iter().any(|w| !set.contains(w)) {
                        if graph.type_of(v).numel() != size {
                            return None;
                        }
                        ends.push(k);
                    }
                }
                if ends.len() == before {
                    return None;
                }
            }
            let main = *ends.iter().max()?;
            let last = sets.iter().flatten().all(|&k| k <= main);
            let read_after = ends.iter().all(|&k| {
                users[nodes[k].output]
                    .iter()
                    .all(|&w| w > main || sets.iter().any(|set| set.contains(&w)))
            });
            (last && read_after).then_some((ends, main))
        };
        let mut best = ends(&sets).map(|e| (sets.clone(), e));
        let mut depends = vec![false; n];
        depends[node.output] = true;
        // Extend regions with later elementwise/reshape nodes reading one region; keep the last valid state.
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            depends[un.output] = un.inputs.iter().any(|&v| depends[v]);
            if region_of[un.output].is_some() {
                continue;
            }
            let mut of: Vec<usize> = un.inputs.iter().filter_map(|&v| region_of[v]).collect();
            of.sort_unstable();
            of.dedup();
            let r = match of.as_slice() {
                [] => continue,
                [r] => *r,
                _ => break,
            };
            let step =
                elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });
            let reads = un
                .inputs
                .iter()
                .all(|&v| region_of[v] == Some(r) || !depends[v]);
            if !(fusible(graph, un) && step && reads && !claimed[u]) {
                break;
            }
            sets[r].push(u);
            region_of[un.output] = Some(r);
            if let Some(e) = ends(&sets) {
                best = Some((sets.clone(), e));
            }
        }
        let Some((sets, (ends, main))) = best else {
            continue;
        };
        let mut members: Vec<usize> = std::iter::once(i)
            .chain(sets.into_iter().flatten())
            .collect();
        // Pull in constant producers that only the members read.
        let mut k = 0;
        while k < members.len() {
            for &v in &nodes[members[k]].inputs {
                let Some(p) = producer[v] else {
                    continue;
                };
                let constant = matches!(
                    nodes[p].primitive,
                    Primitive::Full { .. }
                        | Primitive::Iota { .. }
                        | Primitive::BroadcastInDim { .. }
                        | Primitive::Reshape { .. }
                );
                let only = !output(v) && users[v].iter().all(|w| members.contains(w));
                if constant && only && !depends[v] && !claimed[p] && !members.contains(&p) {
                    members.push(p);
                }
            }
            k += 1;
        }
        members.sort_unstable();
        members.iter().for_each(|&k| claimed[k] = true);
        let extra = ends
            .iter()
            .filter(|&&k| k != main)
            .map(|&k| nodes[k].output)
            .collect();
        found.push(Claim {
            members,
            main,
            extra,
        });
    }
    found
}

/// Assigns each node the root of its loop group (claims first, then joining readers' groups going
/// backwards), and the extra outputs of each root.
fn groups(graph: &Graph) -> (Vec<Option<usize>>, Vec<Vec<Var>>) {
    let nodes = graph.nodes();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        for &v in &node.inputs {
            if !readers[v].contains(&i) {
                readers[v].push(i);
            }
        }
    }
    let mut group: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut extra: Vec<Vec<Var>> = vec![Vec::new(); nodes.len()];
    let mut closed = vec![false; nodes.len()];
    for claim in claims(graph) {
        claim
            .members
            .iter()
            .for_each(|&k| group[k] = Some(claim.main));
        closed[claim.main] = true;
        extra[claim.main] = claim.extra;
    }
    let mut reduces = vec![false; nodes.len()];
    let mut dots = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. });
        if group[i].is_some() || (!fusible(graph, node) && !dot) {
            continue;
        }
        // Join the readers' group when they all share one that isn't a claim and doesn't feed a dot.
        let mut joined = match readers[node.output].split_first() {
            Some((&first, rest)) if !graph.outputs().contains(&node.output) => {
                group[first].filter(|&g| rest.iter().all(|&r| group[r] == Some(g)))
            }
            _ => None,
        };
        let operand = readers[node.output]
            .iter()
            .any(|&r| matches!(nodes[r].primitive, Primitive::DotGeneral { .. }));
        joined = joined.filter(|&g| !operand && !closed[g]);
        // A group holds at most one reduction or dot, whose value must reach the root elementwise.
        let reduction = reduction(&node.primitive);
        if reduction || dot {
            joined = joined.filter(|&g| {
                !reduces[g] && !dots[g] && epilogue(graph, &readers, &group, node.output, g)
            });
        }
        // A dot or reshape that joins nothing stays unfused.
        if joined.is_none() && (dot || matches!(node.primitive, Primitive::Reshape { .. })) {
            continue;
        }
        let g = joined.unwrap_or(i);
        group[i] = Some(g);
        reduces[g] |= reduction;
        dots[g] |= dot;
    }
    (group, extra)
}

/// Whether `p` is a sum or max reduction.
fn reduction(p: &Primitive) -> bool {
    matches!(p, Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. })
}

/// Whether every node of group `g` downstream of value `reduced` is elementwise (no broadcast, layout
/// change or gather), so each output element reads one reduced element.
fn epilogue(
    graph: &Graph,
    readers: &[Vec<usize>],
    group: &[Option<usize>],
    reduced: usize,
    g: usize,
) -> bool {
    let nodes = graph.nodes();
    let mut stack = vec![reduced];
    while let Some(v) = stack.pop() {
        for &r in &readers[v] {
            if group[r] != Some(g) {
                continue;
            }
            let p = &nodes[r].primitive;
            if matches!(
                p,
                Primitive::BroadcastInDim { .. }
                    | Primitive::Transpose { .. }
                    | Primitive::Slice { .. }
                    | Primitive::DynamicSlice { .. }
                    | Primitive::Gather { .. }
                    | Primitive::TakeAlongAxis { .. }
                    | Primitive::Concatenate { .. }
            ) {
                return false;
            }
            stack.push(nodes[r].output);
        }
    }
    true
}
