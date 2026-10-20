use super::diamonds::Row;
use super::{Kernel, MpsModule, horizontal, request};
use crate::DType;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::pass::CompilerPass;
pub(crate) use crate::graph_compiler::{constant, elementwise};

const MAX_INPUTS: usize = 29;

const MAX_AUX: usize = 4;

pub(super) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    let loop_op = matches!(
        node.primitive,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Concatenate { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
    );
    let gather = matches!(node.primitive, Gather { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    let reduction = matches!(node.primitive, ReduceSum { .. } | ReduceMax { .. })
        && graph.type_of(node.inputs[0]).numel() <= u32::MAX as usize;
    let f64 = node
        .inputs
        .iter()
        .chain([&node.output])
        .any(|&v| graph.type_of(v).dtype == DType::F64);
    (loop_op || gather || reduction) && !f64
}

fn is_reduction(node: &Node) -> bool {
    matches!(
        node.primitive,
        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
    )
}

fn expensive(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;
    match node.primitive {
        Exp | Log | Sqrt | Tanh | Logistic => true,
        Div => graph.type_of(node.output).dtype.is_float(),
        _ => false,
    }
}

pub(super) fn fuse(
    graph: &Graph,
    rows: &[Row],
    scalars: &[Var],
    config: &CompilerConfig,
    mut kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
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
    let depends = |v: Var, k: usize| {
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
    };
    let last_read = |v: Var| users[v].iter().copied().max();
    let recomputed = |i: usize| {
        let node = &nodes[i];
        let Some(end) = last_read(node.output) else {
            return false;
        };
        let alive = |v: Var| {
            let mut v = v;
            loop {
                if producer[v].is_none()
                    || constant(graph, &producer, v)
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
    };
    let owned = |u: usize, other: &Row| {
        rows.iter().any(|r| {
            !std::ptr::eq(r, other)
                && (r.root == u || r.inner.contains(&u) || r.reductions.contains(&u))
        })
    };
    let rows: Vec<Row> = rows
        .iter()
        .map(|original| {
            let mut row = original.clone();
            loop {
                let v = nodes[row.root].output;
                let Some(&u) = users[v].first() else { break };
                let inside = |r: usize| r == row.root || row.inner.contains(&r);
                let after = |k: usize| users[nodes[k].output].iter().all(|&r| inside(r) || r >= u);
                let extends = !is_output[v]
                    && !owned(u, original)
                    && fusible[u]
                    && elementwise(&nodes[u].primitive)
                    && users[v][1..].iter().all(|&r| r > u)
                    && row.outputs.iter().all(|&k| after(k))
                    && row.partials.iter().all(|&p| {
                        let v = nodes[u].output;
                        !nodes[u].inputs.iter().any(|&w| w != v && depends(w, p))
                    });
                if !extends {
                    break;
                }
                if users[v].len() > 1 && !recomputed(row.root) {
                    row.outputs.push(row.root);
                }
                row.inner.push(row.root);
                row.root = u;
            }
            row
        })
        .collect();
    let rows = rows.as_slice();
    let row_root = |i: usize| rows.iter().any(|r| r.root == i);
    let mut epilogue: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut ends: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_reduction_epilogue = vec![false; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        let reduction = matches!(
            node.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        );
        let in_row = rows.iter().any(|r| r.reductions.contains(&i));
        if !config.reduction_epilogues || !live[i] || !fusible[i] || !reduction || in_row {
            continue;
        }
        let (mut taken, mut after, mut end) = (vec![i], vec![false; n], None);
        after[node.output] = true;
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }
            let step =
                elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });
            let reads = un
                .inputs
                .iter()
                .all(|&x| after[x] || constant(graph, &producer, x));
            if !fusible[u] || !step || !reads {
                break;
            }
            taken.push(u);
            after[un.output] = true;
            let inside = taken[..taken.len() - 1].iter().all(|&t| {
                let v = nodes[t].output;
                !is_output[v] && users[v].iter().all(|r| taken.contains(r))
            });
            if inside && elementwise(&un.primitive) {
                end = Some(u);
            }
        }
        if let Some(end) = end {
            (epilogue[end], ends[i]) = (Some(i), Some(end));
            for &t in taken.iter().filter(|&&t| t != i && t < end) {
                in_reduction_epilogue[t] = true;
            }
        }
    }
    let mut dot_end: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut dot_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_epilogue = vec![false; nodes.len()];
    let mut dot_aux: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut claimed = vec![false; nodes.len()];
    let mut last_dot: Vec<Option<usize>> = vec![None; n];
    for (i, node) in nodes.iter().enumerate() {
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. }).then_some(i);
        last_dot[node.output] = node
            .inputs
            .iter()
            .map(|&v| last_dot[v])
            .fold(dot, Option::max);
    }
    for (i, node) in nodes.iter().enumerate() {
        let float = |v: Var| {
            matches!(
                graph.type_of(v).dtype,
                DType::F16 | DType::BF16 | DType::F32
            )
        };
        let dot = matches!(node.primitive, Primitive::DotGeneral { .. })
            && node.inputs.iter().all(|&v| float(v))
            && float(node.output);
        if !config.contraction_epilogues || !live[i] || !dot {
            continue;
        }
        let in_row = |u: usize| rows.iter().any(|r| r.root == u || r.inner.contains(&u));
        let cast_by = |un: &Node, grows: bool| {
            let (from, to) = (graph.type_of(un.inputs[0]), graph.type_of(un.output));
            matches!(un.primitive, Primitive::Cast { .. })
                && (to.dtype.size_of() > from.dtype.size_of()) == grows
                && to.dtype.size_of() != from.dtype.size_of()
        };
        let pair: Option<[usize; 2]> = super::codegen::gemm_pair(graph, node)
            .map(|(a, b)| [a.output, b.output].map(|v| producer[v].expect("a slice of the dot")));
        let mut taken = vec![i];
        let (mut after, mut depends) = (vec![false; n], vec![false; n]);
        (after[node.output], depends[node.output]) = (true, true);
        let mut end = None;
        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            depends[un.output] = un.inputs.iter().any(|&v| depends[v]);
            if !live[u] || !un.inputs.iter().any(|&v| after[v]) {
                continue;
            }
            let shape = &graph.type_of(node.output).shape;
            let tupled: Vec<usize> = shape.iter().copied().chain([1]).collect();
            let expands = match un.primitive {
                Primitive::Concatenate { dimension, .. } if dimension + 1 == shape.len() => {
                    un.inputs.iter().all(|&v| graph.type_of(v).shape == *shape)
                }
                Primitive::Concatenate { dimension, .. } if dimension == shape.len() => {
                    un.inputs.iter().all(|&v| graph.type_of(v).shape == tupled)
                }
                _ => false,
            } && un.inputs.len() == 2
                && pair.is_none();
            let step = elementwise(&un.primitive)
                || matches!(un.primitive, Primitive::Reshape { .. })
                || pair.is_some_and(|p| p.contains(&u))
                || expands;
            let fuses = fusible[u] && step && !in_row(u) && !claimed[u];
            let earlier = |v: Var| last_dot[v].is_none_or(|d| d < i);
            let reads = un
                .inputs
                .iter()
                .all(|&v| after[v] || (!depends[v] && earlier(v)));
            if !fuses || !reads {
                break;
            }
            taken.push(u);
            after[un.output] = true;
            let mut kept = vec![u];
            let mut stack = vec![u];
            while let Some(k) = stack.pop() {
                for &v in &nodes[k].inputs {
                    if let Some(p) = producer[v].filter(|p| taken.contains(p) && !kept.contains(p))
                    {
                        kept.push(p);
                        stack.push(p);
                    }
                }
            }
            kept.sort_unstable();
            let outside = |&k: &usize| {
                let v = nodes[k].output;
                is_output[v] || users[v].iter().any(|w| !kept.contains(w))
            };
            let narrowed_aux = |k: usize| {
                let v = nodes[k].output;
                let mut readers = users[v].iter().copied().filter(|w| !kept.contains(w));
                match (readers.next(), readers.next()) {
                    (Some(c), None)
                        if !is_output[v]
                            && matches!(nodes[c].primitive, Primitive::Cast { .. })
                            && graph.type_of(nodes[c].output).dtype.size_of()
                                < graph.type_of(v).dtype.size_of()
                            && live[c]
                            && fusible[c]
                            && !in_row(c)
                            && !claimed[c] =>
                    {
                        c
                    }
                    _ => k,
                }
            };
            let aux: Vec<usize> = kept[..kept.len() - 1]
                .iter()
                .copied()
                .filter(outside)
                .map(narrowed_aux)
                .collect();
            let read_after = aux.iter().all(|&k| {
                users[nodes[k].output]
                    .iter()
                    .all(|&w| kept.contains(&w) || w > u)
            });
            let paired = pair.is_none_or(|p| {
                let half = graph.type_of(nodes[p[0]].output).numel();
                p.iter().all(|k| kept.contains(k))
                    && aux
                        .iter()
                        .all(|&k| graph.type_of(nodes[k].output).numel() == half)
                    && !aux.contains(&i)
                    && graph.type_of(un.output).numel() == half
                    && elementwise(&un.primitive)
            });
            let computed = |v: Var| producer[v].is_some_and(|p| kept.contains(&p));
            let fused_by_readers = |k: usize| {
                let v = nodes[k].output;
                !is_output[v] && users[v].iter().all(|&w| kept.contains(&w) || fusible[w])
            };
            let upcast = kept.iter().any(|&k| {
                cast_by(&nodes[k], true) && computed(nodes[k].inputs[0]) && fused_by_readers(k)
            });
            if kept.first() == Some(&i)
                && read_after
                && aux.len() <= MAX_AUX
                && paired
                && (!expands || aux.is_empty())
                && (!upcast || expands)
            {
                end = Some((kept, aux));
            }
            if expands {
                break;
            }
        }
        if let Some((taken, aux)) = end {
            let end = *taken.last().expect("a node");
            (dot_end[i], dot_of[end]) = (Some(end), Some(i));
            for &k in &taken[1..taken.len() - 1] {
                in_epilogue[k] = true;
            }
            taken.iter().for_each(|&k| claimed[k] = true);
            for k in aux {
                (dot_aux[k], claimed[k]) = (Some(end), true);
            }
        }
    }
    let mut root: Vec<bool> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let users = &users[node.output];
            if dot_end[i].is_some() || in_epilogue[i] || in_reduction_epilogue[i] {
                return false;
            }
            if rows.iter().any(|r| {
                r.inner.contains(&i) || r.reductions.contains(&i) || r.partials.contains(&i)
            }) {
                return false;
            }
            let inside = ends[i].is_some();
            !fusible[i]
                || (is_reduction(node) && !inside)
                || epilogue[i].is_some()
                || dot_of[i].is_some()
                || row_root(i)
                || is_output[node.output]
                || users.iter().any(|&u| !fusible[u])
                || (expensive(graph, node) && users.len() > 1)
        })
        .collect();

    let mut host: Vec<Option<usize>> = dot_aux;
    for (i, node) in nodes.iter().enumerate().rev() {
        let candidate = config.multi_output_fusion
            && live[i]
            && root[i]
            && fusible[i]
            && !is_reduction(node)
            && (is_output[node.output] || expensive(graph, node))
            && dot_of[i].is_none()
            && !rows.iter().any(|r| r.root == i && !r.outputs.is_empty());
        let Some(&first) = users[node.output].first().filter(|_| candidate) else {
            continue;
        };
        let reader = first;
        let mut first = first;
        while !root[first] && users[nodes[first].output].len() == 1 {
            first = users[nodes[first].output][0];
        }
        let fusion = ends[first].unwrap_or(first);
        let after = users[node.output]
            .iter()
            .all(|&u| u == reader || u > fusion);
        if root[fusion]
            && after
            && fusible[first]
            && !row_root(first)
            && dot_of[fusion].is_none()
            && host[fusion].is_none()
            && at_index(graph, &producer, &root, first, node.output)
        {
            host[i] = Some(fusion);
        }
    }

    for row in rows {
        for &k in row.outputs.iter().chain(&row.partials) {
            host[k] = Some(row.root);
        }
    }

    let fusions = loop {
        let fusions: Vec<Option<(Vec<usize>, Vec<Var>)>> = (0..nodes.len())
            .map(|i| {
                let hosted = |p: usize| (root[p] || host[p].is_some()) && host[p] != Some(i);
                let starts: Vec<usize> = std::iter::once(i)
                    .chain((0..nodes.len()).filter(|&h| host[h] == Some(i)))
                    .collect();
                (live[i] && root[i] && fusible[i] && host[i].is_none())
                    .then(|| members(graph, &producer, hosted, &starts))
            })
            .collect();
        let too_big: Vec<usize> = (0..nodes.len())
            .filter(|&i| {
                fusions[i]
                    .as_ref()
                    .is_some_and(|(_, reads)| reads.len() > MAX_INPUTS)
            })
            .collect();
        if too_big.is_empty() {
            break fusions;
        }
        for f in too_big {
            let (members, reads) = fusions[f].as_ref().expect("a fusion");
            assert!(
                members.len() > 1,
                "{}: reads {} values, more than a kernel binds ({MAX_INPUTS})",
                nodes[f].primitive.name(),
                reads.len()
            );
            for &m in members {
                root[m] = true;
            }
            host.iter_mut()
                .filter(|h| **h == Some(f))
                .for_each(|h| *h = None);
        }
    };

    let mut fusions = fusions;
    for d in 0..nodes.len() {
        let dot = matches!(nodes[d].primitive, Primitive::DotGeneral { .. });
        if !config.contraction_epilogues || !dot || !live[d] {
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
        let (members, reads) = members(graph, &producer, hosted, &starts);
        let others: Vec<Var> = starts[1..].iter().map(|&h| nodes[h].output).collect();
        let body = body(graph, &members, &reads, nodes[f].output, &others);
        match reads.len() <= MAX_INPUTS && super::codegen::row_dot(&body).is_some() {
            true => fusions[f] = Some((members, reads)),
            false => (root[p], host) = (true, before),
        }
    }

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
    let alone = |i: usize| match &fusions[i] {
        Some((members, _)) => {
            members.len() == 1 && !matches!(nodes[i].primitive, Primitive::Concatenate { .. })
        }
        None => true,
    };
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
                let label = label(graph, &producer, &shown);
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

pub(crate) fn label(graph: &Graph, producer: &[Option<usize>], members: &[usize]) -> &'static str {
    let nodes = graph.nodes();
    let names = |all: bool| -> Vec<&str> {
        members
            .iter()
            .filter(|&&m| all || !constant(graph, producer, nodes[m].output))
            .map(|&m| graph.label(&nodes[m]))
            .collect()
    };
    let shown = names(false);
    intern(if shown.is_empty() { names(true) } else { shown }.join(FUSION_SEPARATOR))
}

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

fn at_index(graph: &Graph, producer: &[Option<usize>], root: &[bool], r: usize, v: Var) -> bool {
    let nodes = graph.nodes();
    let start = match is_reduction(&nodes[r]) {
        true => nodes[r].inputs[0],
        false => nodes[r].output,
    };
    let mut stack = vec![start];
    while let Some(x) = stack.pop() {
        if x == v {
            return true;
        }
        let Some(p) = producer[x] else { continue };
        if (p == r || !root[p]) && elementwise(&nodes[p].primitive) {
            stack.extend(&nodes[p].inputs);
        }
    }
    false
}

pub(super) struct LoopFusion;

impl<'a> CompilerPass<MpsModule<'a>> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let config = &m.options.config;
        let requested = m.kernels.len();
        let graph = loop {
            m.kernels.truncate(requested);
            let kernels = &mut m.kernels;
            let mut kernel =
                |_: &Graph, by_value: &[bool]| request(kernels, Kernel::Loop(by_value.to_vec()));
            let vars: Vec<Var> = m.scalars.iter().map(|&i| m.graph.inputs()[i]).collect();
            let fused = fuse(&m.graph, &m.rows, &vars, config, &mut kernel);
            let fused = match config.horizontal_fusion {
                true => {
                    let vars: Vec<Var> = m.scalars.iter().map(|&i| fused.inputs()[i]).collect();
                    horizontal::fuse(&fused, &vars, &mut kernel)
                }
                false => fused,
            };
            let unfused = |&i: &usize| {
                let v = fused.inputs()[i];
                let read = |n: &&Node| n.inputs.contains(&v);
                let mut readers = fused.nodes().iter().filter(read);
                readers.any(|n| !matches!(n.primitive, Primitive::Fusion { .. }))
                    || fused.outputs().contains(&v)
            };
            let before = m.scalars.len();
            m.scalars.retain(|i| !unfused(i));
            if m.scalars.len() == before {
                break fused;
            }
        };
        m.graph = graph;
        m.rows.clear();
        Ok(())
    }
}
