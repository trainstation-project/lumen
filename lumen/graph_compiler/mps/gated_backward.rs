//! A gated pair's backward (SwiGLU's, ReGLU's: `z = f(x W1, x W2)`, two
//! dots of one operand combined elementwise, CODA's pairwise epilogue
//! forward, arXiv 2605.19269), its gradients' dots merged as the forward's
//! are: of `x`'s gradient, `g1 W1 + g2 W2` (two dots and their sum, `g1`
//! and `g2` the pair's cotangents) one dot of `[g1 | g2]` and `[W1 | W2]`
//! (contracting both halves: K twice as long), and of the weights', `x^T
//! g1` and `x^T g2` one dot of `x` and `[g1 | g2]`, each weight's its half.
//! `[W1 | W2]` is the concatenate `merge_dots` reads as the forward's block
//! (no copy); `[g1 | g2]` the concatenate the gradient's GEMM writes as its
//! epilogue (`fusion.rs`, `codegen.rs`'s expanding epilogue: each element
//! of `g1` and `g2` computed from the dot's value where it writes it).
//!
//! Matched on dots alone, whatever `f` is: a sum of two dots `dot(g1, a1)`
//! and `dot(g2, a2)` (each read by the sum alone) whose `a1` and `a2` are
//! the operands of two dots of one other operand `x` (the forward pair),
//! and the dots of `g1` and `g2` with `x` (the weights' gradients), if any,
//! of them or their transposes (`g.t() @ x`, a hand-written backward's).

use crate::graph::{Graph, Node, Primitive, Var, intern};

/// A matched pair's backward: the sum's node, its dots' operands (each dot
/// `dot(g, a)` or `dot(a, g)`, `g_first` which), the contracting dimension
/// of each side, and the weights' gradients' dots (`dot(g, x)` or `dot(x,
/// g)`, of `g` or its transpose), if any: theirs, their dot as of `g`, and
/// whether `g` is its lhs.
struct Pair {
    sum: usize,
    dots: [usize; 2],
    g: [Var; 2],
    a: [Var; 2],
    g_first: bool,
    g_dim: usize,
    a_dim: usize,
    weights: Option<([usize; 2], Primitive, bool)>,
}

/// `graph` with each gated pair's backward's dots merged ([`Pair`]).
pub(crate) fn gated_backward(graph: &Graph) -> Graph {
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
    // A plain matmul-form dot: one contracting dimension each, no batch.
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
    // The dots of `a` with another operand: (dot, that operand).
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
    // The dots reading `g`, or its transpose: (dot, its other operand, the
    // dot as of `g`, whether `g` is its lhs).
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
    let mut pairs: Vec<Pair> = Vec::new();
    let mut taken = vec![false; nodes.len()];
    for (s, node) in nodes.iter().enumerate() {
        let (Primitive::Add, &[u, v]) = (&node.primitive, node.inputs.as_slice()) else {
            continue;
        };
        let (Some(e1), Some(e2)) = (producer[u], producer[v]) else {
            continue;
        };
        let alone = |e: usize| users[nodes[e].output] == [s] && !output(nodes[e].output);
        if e1 == e2 || taken[e1] || taken[e2] || !same(e1, e2) || !alone(e1) || !alone(e2) {
            continue;
        }
        let Some((lc, rc)) = dot(e1) else { continue };
        // Which operand is the cotangent: the other is a forward pair's.
        let found = [true, false].into_iter().find_map(|g_first| {
            let (gi, ai) = if g_first { (0, 1) } else { (1, 0) };
            let g = [nodes[e1].inputs[gi], nodes[e2].inputs[gi]];
            let a = [nodes[e1].inputs[ai], nodes[e2].inputs[ai]];
            let ty = |v: Var| graph.type_of(v);
            if g[0] == g[1] || a[0] == a[1] || ty(g[0]) != ty(g[1]) || ty(a[0]) != ty(a[1]) {
                return None;
            }
            // The forward pair: dots of a1 and a2 with one operand x, its
            // only two such (not two of a wider group's, attention's q, k
            // and v: their sum of three is not a pair's).
            let (x, d1, forward) = dots_of(a[0]).into_iter().find_map(|(d1, x)| {
                dots_of(a[1])
                    .into_iter()
                    .find(|&(d2, x2)| x2 == x && same(d1, d2) && d1 != e1 && d2 != e2)
                    .map(|(d2, _)| (x, d1, d1 < d2))
            })?;
            let alike = dots_of(x)
                .into_iter()
                .filter(|&(d, w)| same(d, d1) && ty(w) == ty(a[0]))
                .count();
            if alike != 2 {
                return None;
            }
            // In the forward pair's order (its block's: `[W1 | W2]` as
            // `merge_dots` places it, the weights' gradient's halves so).
            let (dots, g, a) = match forward {
                true => ([e1, e2], g, a),
                false => ([e2, e1], [g[1], g[0]], [a[1], a[0]]),
            };
            // The weights' gradients: the dots of g1 and g2 with x, alike.
            let weights = reads_of(g[0]).into_iter().find_map(|(f1, x1, dot, lhs)| {
                let (f2, ..) = reads_of(g[1])
                    .into_iter()
                    .find(|(_, x2, dot2, lhs2)| *x2 == x && *dot2 == dot && *lhs2 == lhs)?;
                (x1 == x).then_some(([f1, f2], dot, lhs))
            });
            let (g_dim, a_dim) = if g_first { (lc, rc) } else { (rc, lc) };
            Some(Pair {
                sum: s,
                dots,
                g,
                a,
                g_first,
                g_dim,
                a_dim,
                weights,
            })
        });
        if let Some(pair) = found {
            for &k in pair
                .dots
                .iter()
                .chain(pair.weights.iter().flat_map(|w| &w.0))
            {
                taken[k] = true;
            }
            pairs.push(pair);
        }
    }
    if pairs.is_empty() {
        return graph.clone();
    }
    rewrite(graph, &pairs)
}

/// `graph` with `pairs` merged, each node emitted once the values it reads
/// are (a weight's gradient's dot first needs both cotangents).
fn rewrite(graph: &Graph, pairs: &[Pair]) -> Graph {
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<Option<Var>> = vec![None; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = Some(out.input(graph.type_of(v).clone()));
    }
    // The node computing each value, if any.
    let producer = |v: Var| nodes.iter().find(|n| n.output == v);
    // Each pair's `[g1 | g2]` and its weights' gradients' dot, once made.
    let mut stacked: Vec<Option<Var>> = vec![None; pairs.len()];
    let mut weights_dot: Vec<Option<Var>> = vec![None; pairs.len()];
    let role = |i: usize| -> Option<(usize, usize)> {
        pairs.iter().enumerate().find_map(|(k, p)| {
            if p.sum == i {
                Some((k, 0))
            } else if p.dots.contains(&i) {
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
                // The sum's dots: its merged dot.
                Some((_, 1)) => false,
                Some((k, r)) => {
                    let p = &pairs[k];
                    let (Some(g1), Some(g2)) = (map[p.g[0]], map[p.g[1]]) else {
                        return true;
                    };
                    // The merged dots: the lines of the dots (and sum) each
                    // replaces.
                    let merged: Vec<&Node> = match r {
                        0 => vec![node, &nodes[p.dots[0]], &nodes[p.dots[1]]],
                        _ => p
                            .weights
                            .as_ref()
                            .map_or(vec![node], |w| w.0.iter().map(|&f| &nodes[f]).collect()),
                    };
                    // `[g1 | g2]`: the lines computing the cotangents.
                    let stack = *stacked[k].get_or_insert_with(|| {
                        out.set_origins(
                            &p.g.iter().filter_map(|&g| producer(g)).collect::<Vec<_>>(),
                        );
                        let concat = Primitive::Concatenate { dimension: p.g_dim };
                        out.apply(concat, &[g1, g2])
                            .expect("cotangents of one type")
                    });
                    out.set_origins(&merged);
                    if r == 0 {
                        let (Some(a1), Some(a2)) = (map[p.a[0]], map[p.a[1]]) else {
                            return true;
                        };
                        let concat = Primitive::Concatenate { dimension: p.a_dim };
                        let a = out.apply(concat, &[a1, a2]).expect("operands of one type");
                        let ins = if p.g_first { [stack, a] } else { [a, stack] };
                        let merged = out.apply(nodes[p.dots[0]].primitive.clone(), &ins);
                        map[node.output] = Some(merged.expect("the dots' dimension numbers"));
                        return false;
                    }
                    // A weight's gradient: its half of the merged dot.
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
                        // Two dots, named as `merge_dots`' merged ones are.
                        let label = out.label(out.nodes().last().expect("the dot"));
                        out.set_label(v, Some(intern(format!("2x {label}"))));
                        v
                    });
                    // The cotangent's free dimension in the result: the lhs's
                    // first, the rhs's after the lhs's.
                    let shape = out.type_of(merged).shape.clone();
                    let dimension = if g_lhs { 0 } else { x_rank - 1 };
                    let half = shape[dimension] / 2;
                    let (mut starts, mut limits) = (vec![0; shape.len()], shape);
                    (starts[dimension], limits[dimension]) = (half * (r - 2), half * (r - 1));
                    let slice = Primitive::Slice {
                        start_indices: starts,
                        limit_indices: limits,
                    };
                    out.set_origin(node);
                    map[node.output] = Some(out.apply(slice, &[merged]).expect("its half"));
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

/// `dot` reading its lhs's (`lhs`) or rhs's transpose by `permutation` as
/// reading the operand itself: its dimension numbers there mapped, if its
/// free dimensions stay in order (so the result is the same).
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
