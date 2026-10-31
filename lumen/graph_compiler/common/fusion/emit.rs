//! Collects fusion members from roots and hosts and rebuilds the graph with `Fusion` nodes.

use super::analysis::Analysis;
use super::dot_epilogues::DotEpilogues;
use super::reduction_epilogues::ReductionEpilogues;
use super::{Backend, label};
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::common::diamonds::Row;

/// Per root node, its fusion's member nodes and the values it reads, both sorted.
pub(super) type Fusions = Vec<Option<(Vec<usize>, Vec<Var>)>>;

/// Gathers each unhosted root's fusion, splitting fusions that read more than the backend's
/// `max_inputs` values by turning all their members into roots and retrying.
pub(super) fn fusions(
    a: &Analysis,
    backend: &Backend,
    root: &mut [bool],
    host: &mut [Option<usize>],
) -> Fusions {
    let (graph, nodes, producer, live, fusible) =
        (a.graph, a.nodes, &a.producer, &a.live, &a.fusible);

    loop {
        let fusions: Vec<Option<(Vec<usize>, Vec<Var>)>> = (0..nodes.len())
            .map(|i| {
                let hosted = |p: usize| (root[p] || host[p].is_some()) && host[p] != Some(i);

                let starts: Vec<usize> = std::iter::once(i)
                    .chain((0..nodes.len()).filter(|&h| host[h] == Some(i)))
                    .collect();

                (live[i] && root[i] && fusible[i] && host[i].is_none())
                    .then(|| members(graph, producer, hosted, &starts))
            })
            .collect();

        let too_big: Vec<usize> = (0..nodes.len())
            .filter(|&i| {
                fusions[i]
                    .as_ref()
                    .is_some_and(|(_, reads)| reads.len() > backend.max_inputs)
            })
            .collect();

        if too_big.is_empty() {
            break fusions;
        }

        for f in too_big {
            let (members, reads) = fusions[f].as_ref().expect("a fusion");

            assert!(
                members.len() > 1,
                "{}: reads {} values, more than a kernel binds ({})",
                nodes[f].primitive.name(),
                reads.len(),
                backend.max_inputs
            );

            for &m in members {
                root[m] = true;
            }

            host.iter_mut()
                .filter(|h| **h == Some(f))
                .for_each(|h| *h = None);
        }
    }
}

/// Moves a GEMM and its epilogue into the normalization row fusion that reads its result, when the
/// row kernel can compute the dot itself; reverts the move if the resulting body doesn't fit.
#[allow(clippy::too_many_arguments)]
pub(super) fn row_dots(
    a: &Analysis,
    rows: &[Row],
    backend: &Backend,
    dots: &DotEpilogues,
    root: &mut [bool],
    host: &mut Vec<Option<usize>>,
    fusions: &mut Fusions,
) {
    let (graph, nodes, producer, is_output, live) =
        (a.graph, a.nodes, &a.producer, &a.is_output, &a.live);

    let dot_end = &dots.end;
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);

    for d in 0..nodes.len() {
        let dot = matches!(nodes[d].primitive, Primitive::DotGeneral { .. });

        if !backend.contraction_epilogues || !dot || !live[d] {
            continue;
        }

        let p = dot_end[d].unwrap_or(d);
        let v = nodes[p].output;

        if !root[p] || host[p].is_some() {
            continue;
        }

        let Some(f) = (0..nodes.len()).find(|&f| {
            fusions[f]
                .as_ref()
                .is_some_and(|(_, reads)| reads.contains(&v))
                && row_root(f)
        }) else {
            continue;
        };

        let read_by = |w: Var| -> Vec<usize> {
            (0..nodes.len())
                .filter(|&i| live[i] && root[i] && host[i].is_none())
                .filter(|&i| match &fusions[i] {
                    Some((_, reads)) => reads.contains(&w),
                    None => nodes[i].inputs.contains(&w),
                })
                .collect()
        };

        let written: Vec<Var> = std::iter::once(v)
            .chain(
                (0..nodes.len())
                    .filter(|&h| host[h] == Some(p))
                    .map(|h| nodes[h].output),
            )
            .collect();

        // Every value the dot fusion writes must be read only at or after the row fusion.
        if !written.iter().all(|&w| read_by(w).iter().all(|&i| i >= f)) {
            continue;
        }

        let elsewhere = is_output[v] || read_by(v).iter().any(|&i| i != f);
        let before = host.clone();
        root[p] = false;

        for h in host.iter_mut().filter(|h| **h == Some(p)) {
            *h = Some(f);
        }

        if elsewhere {
            host[p] = Some(f);
        }

        let hosted = |q: usize| (root[q] || host[q].is_some()) && host[q] != Some(f);

        let starts: Vec<usize> = std::iter::once(f)
            .chain((0..nodes.len()).filter(|&h| host[h] == Some(f)))
            .collect();

        let (members, reads) = members(graph, producer, hosted, &starts);
        let others: Vec<Var> = starts[1..].iter().map(|&h| nodes[h].output).collect();
        let body = body(graph, &members, &reads, nodes[f].output, &others);

        match reads.len() <= backend.max_inputs && (backend.row_dot)(&body) {
            true => fusions[f] = Some((members, reads)),
            false => (root[p], *host) = (true, before),
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// Emits the fused graph, ordering steps so each runs after the values it reads, with hosted values
/// exposed through `FusionOutput` nodes.
pub(super) fn emitted(
    a: &Analysis,
    scalars: &[Var],
    root: &[bool],
    host: &[Option<usize>],
    fusions: &Fusions,
    dots: &DotEpilogues,
    reductions: &ReductionEpilogues,
    mut kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    let (graph, nodes, n) = (a.graph, a.nodes, a.n);
    let (producer, live) = (&a.producer, &a.live);
    let (dot_of, epilogue) = (&dots.of, &reductions.epilogue);
    let mut fused = Graph::new();
    let mut var: Vec<Var> = vec![usize::MAX; n];

    for &v in graph.inputs() {
        var[v] = fused.input(graph.type_of(v).clone());
    }

    let mut hosts: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for (h, &f) in host.iter().enumerate() {
        if let Some(f) = f {
            hosts[f].push(h);
        }
    }

    // Single-node fusions other than concatenations are emitted as the plain node.
    let alone = |i: usize| match &fusions[i] {
        Some((members, _)) => {
            members.len() == 1 && !matches!(nodes[i].primitive, Primitive::Concatenate { .. })
        }
        None => true,
    };

    // Topological order over steps, since hosting can make a fusion read values produced later in node order.
    let order = {
        let mut ready = vec![false; n];
        graph.inputs().iter().for_each(|&v| ready[v] = true);
        let step = |i: usize| live[i] && root[i] && host[i].is_none();
        let (mut order, mut waiting) = (Vec::new(), Vec::new());

        for i in 0..nodes.len() {
            waiting.push(i);

            while let Some(k) = waiting.iter().position(|&w| {
                let reads = match &fusions[w] {
                    Some((_, reads)) if !alone(w) => reads,
                    _ => &nodes[w].inputs,
                };

                !step(w) || reads.iter().all(|&v| ready[v])
            }) {
                let w = waiting.remove(k);

                if step(w) {
                    for &h in std::iter::once(&w).chain(&hosts[w]) {
                        ready[nodes[h].output] = true;
                    }
                }

                order.push(w);
            }
        }

        assert!(
            waiting.is_empty(),
            "every step reads values computed before it"
        );

        order
    };

    for i in order {
        let node = &nodes[i];
        let main = dot_of[i].or(epilogue[i]).unwrap_or(i);
        fused.set_origin(&nodes[main]);

        if !live[i] || !root[i] || host[i].is_some() {
            continue;
        }

        let out = match &fusions[i] {
            Some((members, reads)) if !alone(i) => {
                let hosted: Vec<Var> = hosts[i].iter().map(|&h| nodes[h].output).collect();
                let body = body(graph, members, reads, node.output, &hosted);
                let by_value: Vec<bool> = reads.iter().map(|v| scalars.contains(v)).collect();
                let name = kernel(&body, &by_value);

                // Hide the slices and reshapes that split a paired GEMM's output from the label, and list the dot first.
                let of_dot = |v: Var, layout: &dyn Fn(&Primitive) -> bool| {
                    producer[v].is_some_and(|d| members.contains(&d) && layout(&nodes[d].primitive))
                };

                let dot = |p: &Primitive| matches!(p, Primitive::DotGeneral { .. });
                let pairs = |p: &Primitive| matches!(p, Primitive::Reshape { .. });

                let half = |m: &&usize| {
                    let node = &nodes[**m];
                    let input = node.inputs.first().copied();

                    match node.primitive {
                        Primitive::Slice { .. } => input.is_some_and(|v| {
                            of_dot(v, &dot)
                                || producer[v].is_some_and(|r| {
                                    pairs(&nodes[r].primitive) && of_dot(nodes[r].inputs[0], &dot)
                                })
                        }),
                        Primitive::Reshape { .. } => input.is_some_and(|v| {
                            of_dot(v, &dot)
                                || producer[v].is_some_and(|s| {
                                    matches!(nodes[s].primitive, Primitive::Slice { .. })
                                        && producer[nodes[s].inputs[0]].is_some_and(|r| {
                                            pairs(&nodes[r].primitive)
                                                && of_dot(nodes[r].inputs[0], &dot)
                                        })
                                })
                        }),
                        _ => false,
                    }
                };

                let mut shown: Vec<usize> = members.iter().filter(|m| !half(m)).copied().collect();

                if let Some(d) = dot_of[i] {
                    shown.sort_by_key(|&m| m != d);
                }

                let label = label(graph, producer, &shown);
                let reads: Vec<Var> = reads.iter().map(|&v| var[v]).collect();
                let out = fused.apply(Primitive::Fusion { name, label, body }, &reads);
                let out = out.expect("a fused graph is typed as the original");

                for (k, &v) in hosted.iter().enumerate() {
                    let ty = graph.type_of(v).clone();
                    let output = Primitive::FusionOutput { index: k + 1, ty };
                    var[v] = fused.apply(output, &[out]).expect("the fusion's output");
                }

                Ok(out)
            }
            _ => {
                let inputs: Vec<Var> = node.inputs.iter().map(|&v| var[v]).collect();
                let out = fused.apply(node.primitive.clone(), &inputs);

                if let Ok(v) = out {
                    fused.set_label(v, node.label);
                }

                out
            }
        };

        var[node.output] = out.expect("a fused graph is typed as the original");
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| var[v]).collect();

    fused
        .set_outputs(&outputs)
        .expect("outputs are values of the fused graph");

    fused
}

/// Nodes reachable from `starts` through non-root producers, and the values read from outside them.
fn members(
    graph: &Graph,
    producer: &[Option<usize>],
    root: impl Fn(usize) -> bool,
    starts: &[usize],
) -> (Vec<usize>, Vec<Var>) {
    let nodes = graph.nodes();
    let (mut members, mut reads) = (starts.to_vec(), Vec::new());
    let mut stack = starts.to_vec();

    while let Some(i) = stack.pop() {
        for &v in &nodes[i].inputs {
            match producer[v] {
                Some(p) if !root(p) => {
                    if !members.contains(&p) {
                        members.push(p);
                        stack.push(p);
                    }
                }
                _ => {
                    if !reads.contains(&v) {
                        reads.push(v);
                    }
                }
            }
        }
    }

    members.sort_unstable();
    reads.sort_unstable();
    (members, reads)
}

/// Builds a fusion body over `members` with `reads` as inputs and `root` then `hosted` as outputs.
fn body(graph: &Graph, members: &[usize], reads: &[Var], root: Var, hosted: &[Var]) -> Graph {
    let mut body = Graph::new();
    let mut var = std::collections::HashMap::new();

    for &v in reads {
        var.insert(v, body.input(graph.type_of(v).clone()));
    }

    for &i in members {
        let node = &graph.nodes()[i];
        let inputs: Vec<Var> = node.inputs.iter().map(|v| var[v]).collect();
        body.set_origin(node);

        let out = body
            .apply(node.primitive.clone(), &inputs)
            .expect("a fusion body is typed as the original");

        var.insert(node.output, out);
    }

    let outputs: Vec<Var> = std::iter::once(root)
        .chain(hosted.iter().copied())
        .map(|v| var[&v])
        .collect();

    body.set_outputs(&outputs)
        .expect("the root and hosted values are values of the body");

    body
}
