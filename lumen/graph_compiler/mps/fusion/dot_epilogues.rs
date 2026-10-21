use super::analysis::Analysis;
use super::{MAX_AUX, elementwise};
use crate::DType;
use crate::graph::{Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::common::diamonds::Row;

pub(super) struct DotEpilogues {
    pub(super) end: Vec<Option<usize>>,
    pub(super) of: Vec<Option<usize>>,
    pub(super) inside: Vec<bool>,
    pub(super) aux: Vec<Option<usize>>,
}

pub(super) fn dot_epilogues(a: &Analysis, rows: &[Row], config: &CompilerConfig) -> DotEpilogues {
    let (graph, nodes, n) = (a.graph, a.nodes, a.n);
    let (producer, is_output, live, users, fusible) =
        (&a.producer, &a.is_output, &a.live, &a.users, &a.fusible);
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
        if let Some(regions) = crate::codegen::mps::gemm_regions(graph, node) {
            let index_of = |v: Var| producer[v].expect("a node's value");
            let mut sets: Vec<Vec<usize>> = regions
                .iter()
                .map(|c| {
                    let mut set = Vec::new();
                    for v in c.members() {
                        let mut k = index_of(v);
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
                    set.sort_unstable();
                    set
                })
                .collect();
            let mut region_of: Vec<Option<usize>> = vec![None; n];
            for (r, set) in sets.iter().enumerate() {
                for &k in set {
                    region_of[nodes[k].output] = Some(r);
                }
            }
            let mut depends = vec![false; n];
            depends[node.output] = true;
            for (u, un) in nodes.iter().enumerate().skip(i + 1) {
                depends[un.output] = un.inputs.iter().any(|&v| depends[v]);
                if !live[u] || region_of[un.output].is_some() {
                    continue;
                }
                let mut of: Vec<usize> = un.inputs.iter().filter_map(|&v| region_of[v]).collect();
                of.sort_unstable();
                of.dedup();
                let [r] = of.as_slice() else {
                    continue;
                };
                let earlier = |v: Var| last_dot[v].is_none_or(|d| d < i);
                let steps =
                    elementwise(&un.primitive) || matches!(un.primitive, Primitive::Reshape { .. });
                let reads = un
                    .inputs
                    .iter()
                    .all(|&v| region_of[v] == Some(*r) || (!depends[v] && earlier(v)));
                if fusible[u] && steps && reads && !in_row(u) && !claimed[u] {
                    sets[*r].push(u);
                    region_of[un.output] = Some(*r);
                }
            }
            let ends: Option<Vec<usize>> = regions
                .iter()
                .zip(&sets)
                .map(|(c, set)| {
                    let outside: Vec<usize> = set
                        .iter()
                        .copied()
                        .filter(|&k| {
                            let v = nodes[k].output;
                            is_output[v] || users[v].iter().any(|w| !set.contains(w))
                        })
                        .collect();
                    match outside.as_slice() {
                        [end]
                            if graph.type_of(nodes[*end].output).numel()
                                == graph.type_of(node.output).numel()
                                    / graph
                                        .type_of(node.output)
                                        .shape
                                        .last()
                                        .copied()
                                        .unwrap_or(1)
                                    * c.size() =>
                        {
                            Some(*end)
                        }
                        _ => None,
                    }
                })
                .collect();
            if let Some(ends) = ends.filter(|e| e.len() - 1 <= MAX_AUX) {
                let main = *ends.iter().max().expect("regions");
                (dot_end[i], dot_of[main]) = (Some(main), Some(i));
                claimed[i] = true;
                for set in &sets {
                    for &k in set {
                        claimed[k] = true;
                        if k != main {
                            in_epilogue[k] = true;
                        }
                    }
                }
                for &end in ends.iter().filter(|&&e| e != main) {
                    dot_aux[end] = Some(main);
                }
                continue;
            }
        }
        let pair: Option<[usize; 2]> = crate::codegen::mps::gemm_pair(graph, node)
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
    DotEpilogues {
        end: dot_end,
        of: dot_of,
        inside: in_epilogue,
        aux: dot_aux,
    }
}
