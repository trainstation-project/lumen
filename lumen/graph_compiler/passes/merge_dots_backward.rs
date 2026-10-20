use crate::graph::{Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::{CompilerPass, Module};

struct Group {
    sum: usize,
    adds: Vec<usize>,
    dots: Vec<usize>,
    g: Vec<Var>,
    a: Vec<Var>,
    g_first: bool,
    g_dim: usize,
    a_dim: usize,
    weights: Option<(Vec<usize>, Primitive, bool)>,
}

fn merge_dots_backward(graph: &Graph) -> Graph {
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
    let dot = |i: usize| match &nodes[i].primitive {
        Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            lhs_batch,
            ..
        } => (lhs_batch.is_empty() && lhs_contracting.len() == 1 && rhs_contracting.len() == 1)
            .then(|| (lhs_contracting[0], rhs_contracting[0])),
        _ => None,
    };
    let same = |a: usize, b: usize| nodes[a].primitive == nodes[b].primitive;
    let weight = |mut v: Var| loop {
        match producer[v] {
            None => break graph.inputs().contains(&v),
            Some(k) if nodes[k].inputs.len() == 1 && elementwise(&nodes[k].primitive) => {
                v = nodes[k].inputs[0]
            }
            Some(_) => break false,
        }
    };
    let dots_of = |a: Var| -> Vec<(usize, Var)> {
        users[a]
            .iter()
            .filter(|&&u| dot(u).is_some())
            .filter_map(|&u| {
                let ins = &nodes[u].inputs;
                match (ins[0] == a, ins[1] == a) {
                    (true, false) => Some((u, ins[1])),
                    (false, true) => Some((u, ins[0])),
                    _ => None,
                }
            })
            .collect()
    };
    let reads_of = |g: Var| -> Vec<(usize, Var, Primitive, bool)> {
        let transposes = users[g].iter().filter_map(|&t| match &nodes[t].primitive {
            Primitive::Transpose { permutation } => Some((nodes[t].output, Some(permutation))),
            _ => None,
        });
        std::iter::once((g, None))
            .chain(transposes)
            .flat_map(|(u, permutation)| {
                dots_of(u).into_iter().filter_map(move |(d, other)| {
                    let lhs = nodes[d].inputs[0] == u;
                    let dot = match permutation {
                        None => nodes[d].primitive.clone(),
                        Some(p) => through(&nodes[d].primitive, lhs, p)?,
                    };
                    Some((d, other, dot, lhs))
                })
            })
            .collect()
    };
    let mut groups: Vec<Group> = Vec::new();
    let mut taken = vec![false; nodes.len()];
    for s in (0..nodes.len()).rev() {
        if taken[s] || !matches!(nodes[s].primitive, Primitive::Add) {
            continue;
        }
        let alone =
            |e: usize, by: usize| users[nodes[e].output] == [by] && !output(nodes[e].output);
        let (mut adds, mut terms) = (vec![s], Vec::new());
        let mut k = 0;
        let mut sum = true;
        while sum && k < adds.len() {
            for &u in &nodes[adds[k]].inputs {
                match producer[u] {
                    Some(e) if alone(e, adds[k]) && !taken[e] => match nodes[e].primitive {
                        Primitive::Add => adds.push(e),
                        _ if dot(e).is_some() => terms.push(e),
                        _ => sum = false,
                    },
                    _ => sum = false,
                }
            }
            k += 1;
        }
        if !sum || terms.len() < 2 || terms.iter().any(|&e| !same(e, terms[0])) {
            continue;
        }
        let Some((lc, rc)) = dot(terms[0]) else {
            continue;
        };
        let found = [true, false].into_iter().find_map(|g_first| {
            let (gi, ai) = if g_first { (0, 1) } else { (1, 0) };
            let g: Vec<Var> = terms.iter().map(|&e| nodes[e].inputs[gi]).collect();
            let a: Vec<Var> = terms.iter().map(|&e| nodes[e].inputs[ai]).collect();
            let ty = |v: Var| graph.type_of(v);
            let distinct = |vs: &[Var]| (1..vs.len()).all(|j| !vs[..j].contains(&vs[j]));
            let one_type = |vs: &[Var]| vs.iter().all(|&v| ty(v) == ty(vs[0]));
            let of_weights = a.iter().all(|&v| weight(v));
            if !distinct(&g) || !distinct(&a) || !one_type(&g) || !one_type(&a) || !of_weights {
                return None;
            }
            let (x, forward) = dots_of(a[0]).into_iter().find_map(|(d1, x)| {
                let forward: Vec<usize> = a
                    .iter()
                    .map(|&ai| {
                        dots_of(ai)
                            .into_iter()
                            .find(|&(d, x2)| x2 == x && same(d1, d) && !terms.contains(&d))
                            .map(|(d, _)| d)
                    })
                    .collect::<Option<_>>()?;
                Some((x, forward))
            })?;
            let alike = dots_of(x)
                .into_iter()
                .filter(|&(d, w)| same(d, forward[0]) && ty(w) == ty(a[0]))
                .count();
            if alike != terms.len() {
                return None;
            }
            let mut order: Vec<usize> = (0..terms.len()).collect();
            order.sort_by_key(|&j| forward[j]);
            let dots: Vec<usize> = order.iter().map(|&j| terms[j]).collect();
            let g: Vec<Var> = order.iter().map(|&j| g[j]).collect();
            let a: Vec<Var> = order.iter().map(|&j| a[j]).collect();
            let weights = reads_of(g[0]).into_iter().find_map(|(f1, x1, dot, lhs)| {
                let mut fs = vec![f1];
                for &gi in &g[1..] {
                    let (f, ..) = reads_of(gi)
                        .into_iter()
                        .find(|(_, x2, dot2, lhs2)| *x2 == x && *dot2 == dot && *lhs2 == lhs)?;
                    fs.push(f);
                }
                (x1 == x).then_some((fs, dot, lhs))
            });
            let (g_dim, a_dim) = if g_first { (lc, rc) } else { (rc, lc) };
            Some(Group {
                sum: s,
                adds: adds[1..].to_vec(),
                dots,
                g,
                a,
                g_first,
                g_dim,
                a_dim,
                weights,
            })
        });
        if let Some(group) = found {
            for &k in group
                .adds
                .iter()
                .chain(&group.dots)
                .chain(group.weights.iter().flat_map(|w| &w.0))
            {
                taken[k] = true;
            }
            groups.push(group);
        }
    }
    if groups.is_empty() {
        return graph.clone();
    }
    rewrite(graph, &groups)
}

fn rewrite(graph: &Graph, groups: &[Group]) -> Graph {
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<Option<Var>> = vec![None; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = Some(out.input(graph.type_of(v).clone()));
    }
    let producer = |v: Var| nodes.iter().find(|n| n.output == v);
    let mut stacked: Vec<Option<Var>> = vec![None; groups.len()];
    let mut weights_dot: Vec<Option<Var>> = vec![None; groups.len()];
    let role = |i: usize| -> Option<(usize, usize)> {
        groups.iter().enumerate().find_map(|(k, p)| {
            if p.sum == i {
                Some((k, 0))
            } else if p.dots.contains(&i) || p.adds.contains(&i) {
                Some((k, 1))
            } else {
                p.weights
                    .as_ref()
                    .and_then(|w| w.0.iter().position(|&f| f == i).map(|h| (k, 2 + h)))
            }
        })
    };
    let mut pending: Vec<usize> = (0..nodes.len()).collect();
    loop {
        let before = pending.len();
        pending.retain(|&i| {
            let node: &Node = &nodes[i];
            out.set_origin(node);
            match role(i) {
                Some((_, 1)) => false,
                Some((k, r)) => {
                    let p = &groups[k];
                    let Some(g) = p.g.iter().map(|&g| map[g]).collect::<Option<Vec<_>>>() else {
                        return true;
                    };
                    let merged: Vec<&Node> = match r {
                        0 => std::iter::once(node)
                            .chain(p.adds.iter().chain(&p.dots).map(|&e| &nodes[e]))
                            .collect(),
                        _ => p
                            .weights
                            .as_ref()
                            .map_or(vec![node], |w| w.0.iter().map(|&f| &nodes[f]).collect()),
                    };
                    let stack = *stacked[k].get_or_insert_with(|| {
                        out.set_origins(
                            &p.g.iter().filter_map(|&g| producer(g)).collect::<Vec<_>>(),
                        );
                        let concat = Primitive::Concatenate {
                            dimension: p.g_dim,
                            in_place: false,
                        };
                        out.apply(concat, &g).expect("cotangents of one type")
                    });
                    out.set_origins(&merged);
                    if r == 0 {
                        let Some(a) = p.a.iter().map(|&a| map[a]).collect::<Option<Vec<_>>>()
                        else {
                            return true;
                        };
                        let concat = Primitive::Concatenate {
                            dimension: p.a_dim,
                            in_place: false,
                        };
                        let a = out.apply(concat, &a).expect("operands of one type");
                        let ins = if p.g_first { [stack, a] } else { [a, stack] };
                        let merged = out.apply(nodes[p.dots[0]].primitive.clone(), &ins);
                        map[node.output] = Some(merged.expect("the dots' dimension numbers"));
                        return false;
                    }
                    let (w, dot, g_lhs) = p.weights.as_ref().expect("a weight's gradient");
                    let (f, g_lhs) = (&nodes[w[0]], *g_lhs);
                    let x_rank = graph.type_of(f.inputs[usize::from(g_lhs)]).shape.len();
                    let Some(x) = map[f.inputs[usize::from(g_lhs)]] else {
                        return true;
                    };
                    let merged = *weights_dot[k].get_or_insert_with(|| {
                        let ins = if g_lhs { [stack, x] } else { [x, stack] };
                        let v = out
                            .apply(dot.clone(), &ins)
                            .expect("the weights' gradients' dimension numbers");
                        let label = out.label(out.nodes().last().expect("the dot"));
                        let n = p.g.len();
                        out.set_label(v, Some(intern(format!("{n}x {label}"))));
                        v
                    });
                    let shape = out.type_of(merged).shape.clone();
                    let dimension = if g_lhs { 0 } else { x_rank - 1 };
                    let part = shape[dimension] / p.g.len();
                    let (mut starts, mut limits) = (vec![0; shape.len()], shape);
                    (starts[dimension], limits[dimension]) = (part * (r - 2), part * (r - 1));
                    let slice = Primitive::Slice {
                        start_indices: starts,
                        limit_indices: limits,
                        view: false,
                    };
                    out.set_origin(node);
                    map[node.output] = Some(out.apply(slice, &[merged]).expect("its part"));
                    false
                }
                None => {
                    let Some(inputs) = node
                        .inputs
                        .iter()
                        .map(|&v| map[v])
                        .collect::<Option<Vec<_>>>()
                    else {
                        return true;
                    };
                    let v = out
                        .apply(node.primitive.clone(), &inputs)
                        .expect("a graph is typed as the original");
                    out.set_label(v, node.label);
                    map[node.output] = Some(v);
                    false
                }
            }
        });
        if pending.is_empty() || pending.len() == before {
            break;
        }
    }
    assert!(
        pending.is_empty(),
        "every node's values are computed before it"
    );
    let outputs: Vec<Var> = graph
        .outputs()
        .iter()
        .map(|&v| map[v].expect("an output's value"))
        .collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

fn through(dot: &Primitive, lhs: bool, permutation: &[usize]) -> Option<Primitive> {
    let Primitive::DotGeneral {
        mut lhs_contracting,
        mut rhs_contracting,
        mut lhs_batch,
        mut rhs_batch,
        accum_dtype,
        output_dtype,
    } = dot.clone()
    else {
        return None;
    };
    let (contracting, batch) = match lhs {
        true => (&mut lhs_contracting, &mut lhs_batch),
        false => (&mut rhs_contracting, &mut rhs_batch),
    };
    let free: Vec<usize> = (0..permutation.len())
        .filter(|d| !contracting.contains(d) && !batch.contains(d))
        .map(|d| permutation[d])
        .collect();
    if !free.is_sorted() {
        return None;
    }
    for d in contracting.iter_mut().chain(batch.iter_mut()) {
        *d = permutation[*d];
    }
    Some(Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        accum_dtype,
        output_dtype,
    })
}

pub(crate) struct MergeDotsBackward;

impl<M: Module> CompilerPass<M> for MergeDotsBackward {
    fn name(&self) -> &'static str {
        "merge_dots_backward"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = merge_dots_backward(m.graph());
        Ok(())
    }
}
