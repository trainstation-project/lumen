//! Finds elementwise epilogues fused into GEMM kernels, including split-column regions and auxiliary outputs.

use super::analysis::Analysis;
use super::{Backend, elementwise};
use crate::DType;
use crate::graph::{Node, Primitive, Var};
use crate::graph_compiler::passes::diamonds::Row;

/// GEMM epilogue assignment, indexed by node.
pub(super) struct DotEpilogues {
    /// For a dot, the last node of its epilogue.
    pub(super) end: Vec<Option<usize>>,
    /// For the last node of an epilogue, its dot.
    pub(super) of: Vec<Option<usize>>,
    /// Whether a node sits strictly inside some dot's epilogue.
    pub(super) inside: Vec<bool>,
    /// For an extra value the epilogue writes, the epilogue end hosting it.
    pub(super) aux: Vec<Option<usize>>,
    /// Updates (`dynamic_update_slice`s) an epilogue writes its region's values through, into the slot of the
    /// operand they update: fusible only as an epilogue's.
    pub(super) updates: Vec<usize>,
}

/// Assigns each float GEMM the longest elementwise chain after it that can run in its epilogue; with
/// `combine_epilogues`, a split-K combine (`combine`) too, as its dot's GEMM would unsplit.
pub(super) fn dot_epilogues(a: &Analysis, rows: &[Row], backend: &Backend) -> DotEpilogues {
    let (graph, nodes, n) = (a.graph, a.nodes, a.n);

    let (producer, is_output, live, users, fusible) =
        (&a.producer, &a.is_output, &a.live, &a.users, &a.fusible);

    let mut dot_end: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut dot_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut in_epilogue = vec![false; nodes.len()];
    let mut dot_aux: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut claimed = vec![false; nodes.len()];
    let mut updates = Vec::new();

    // Whether `un` writes `v` (an epilogue's value) into the slot of an operand computed apart from it: an
    // update the epilogue stores straight into that operand, with `update_epilogues`.
    let update = |un: &Node, v: Var| {
        backend.update_epilogues
            && matches!(un.primitive, Primitive::DynamicUpdateSlice)
            && un.inputs[1] == v
            && un.inputs[0] != v
    };

    // Latest dot each value depends on, so epilogues only read operands finished before their dot.
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
            && float(node.output)
            && (backend.dot_epilogue)(graph, node);

        let combine = backend.combine_epilogues && combine(a, i);

        if !backend.contraction_epilogues || !live[i] || !(dot || combine) {
            continue;
        }

        // The value the epilogue starts from and the host's own nodes: a dot's product, or a combine's sum
        // through the cast rounding it (its dot's output dtype, unsplit).
        let (value, own) = match combine {
            true => {
                let value = crate::codegen::regions::combine_value(graph, node);
                let cast = producer[value].filter(|&c| c != i);
                (value, std::iter::once(i).chain(cast).collect::<Vec<_>>())
            }
            false => (node.output, vec![i]),
        };

        let in_row = |u: usize| rows.iter().any(|r| r.root == u || r.inner.contains(&u));

        let cast_by = |un: &Node, grows: bool| {
            let (from, to) = (graph.type_of(un.inputs[0]), graph.type_of(un.output));

            matches!(un.primitive, Primitive::Cast { .. })
                && (to.dtype.size_of() > from.dtype.size_of()) == grows
                && to.dtype.size_of() != from.dtype.size_of()
        };

        let pair = crate::codegen::regions::gemm_pair(graph, node);

        let regions = match combine {
            true => crate::codegen::regions::combine_regions(graph, value),
            false => crate::codegen::regions::gemm_regions(graph, node).filter(|_| pair.is_none()),
        };

        // When readers split the output into column regions, grow each region separately and
        // keep the largest assignment whose region outputs all fit.
        if let Some(regions) = regions {
            let index_of = |v: Var| producer[v].expect("a node's value");

            // Seed each region with its slicing nodes back to the dot.
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
                                Some(p) if !own.contains(&p) => k = p,
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

            // Ends of every region and the main (last) one, when each region writes values of its own size.
            let fused = |sets: &[Vec<usize>]| -> Option<(Vec<usize>, usize)> {
                let mut ends = Vec::new();

                for (c, set) in regions.iter().zip(sets) {
                    let size = graph.type_of(value).numel() / graph.type_of(value).shape[c.axis]
                        * c.size();

                    let outside = set.iter().copied().filter(|&k| {
                        let v = nodes[k].output;
                        is_output[v] || users[v].iter().any(|w| !set.contains(w))
                    });

                    let before = ends.len();

                    for k in outside {
                        // An update writes the region's values into its operand's slot.
                        let written = match nodes[k].primitive {
                            Primitive::DynamicUpdateSlice => nodes[k].inputs[1],
                            _ => nodes[k].output,
                        };

                        if graph.type_of(written).numel() != size {
                            return None;
                        }

                        ends.push(k);
                    }

                    if ends.len() == before {
                        return None;
                    }
                }

                // A combine's value read whole besides its regions (prefill attention reading a merged
                // projection's values as a view) is stored too, by its own node (the rounding cast).
                let whole = is_output[value]
                    || users[value]
                        .iter()
                        .any(|w| !sets.iter().any(|set| set.contains(w)));

                if whole {
                    ends.push(producer[value].filter(|&c| combine && c != i)?);
                }

                let main = *ends.iter().max()?;

                let read_after = ends.iter().all(|&k| {
                    users[nodes[k].output]
                        .iter()
                        .all(|&w| w > main || sets.iter().any(|set| set.contains(&w)))
                });

                (ends.len() - 1 <= backend.max_aux && read_after).then_some((ends, main))
            };

            let mut best = fused(&sets).map(|f| (sets.clone(), f));
            let mut depends = vec![false; n];
            depends[node.output] = true;
            depends[value] = true;

            for (u, un) in nodes.iter().enumerate().skip(i + 1) {
                depends[un.output] = un.inputs.iter().any(|&v| depends[v]);

                if !live[u] || region_of[un.output].is_some() {
                    continue;
                }

                let mut of: Vec<usize> = un.inputs.iter().filter_map(|&v| region_of[v]).collect();
                of.sort_unstable();
                of.dedup();

                let r = match of.as_slice() {
                    [] => continue,
                    [r] => r,
                    _ => break,
                };

                let earlier = |v: Var| last_dot[v].is_none_or(|d| d < i);

                let updates_slot =
                    (un.inputs.get(1)).is_some_and(|&v| region_of[v] == Some(*r) && update(un, v));

                let steps = elementwise(&un.primitive)
                    || matches!(un.primitive, Primitive::Reshape { .. })
                    || (backend.swap_epilogues && matches!(un.primitive, Primitive::SwapPairs))
                    || updates_slot;

                let reads = un
                    .inputs
                    .iter()
                    .all(|&v| region_of[v] == Some(*r) || (!depends[v] && earlier(v)));

                if !((fusible[u] || updates_slot) && steps && reads && !in_row(u) && !claimed[u]) {
                    break;
                }

                sets[*r].push(u);
                region_of[un.output] = Some(*r);

                if let Some(f) = fused(&sets) {
                    best = Some((sets.clone(), f));
                }
            }

            if let Some((sets, (ends, main))) = best {
                (dot_end[i], dot_of[main]) = (Some(main), Some(i));

                for &k in &own {
                    claimed[k] = true;
                    in_epilogue[k] = k != i;
                }

                updates.extend(
                    sets.iter()
                        .flatten()
                        .filter(|&&k| matches!(nodes[k].primitive, Primitive::DynamicUpdateSlice)),
                );

                for &k in sets.iter().flatten() {
                    claimed[k] = true;

                    if k != main {
                        in_epilogue[k] = true;
                    }
                }

                for &end in ends.iter().filter(|&&e| e != main) {
                    dot_aux[end] = Some(main);
                }

                continue;
            }
        }

        // Otherwise grow a single chain, recording the longest prefix that fits as the epilogue.
        let pair: Option<[usize; 2]> = pair
            .map(|(a, b)| [a.output, b.output].map(|v| producer[v].expect("a slice of the dot")));

        let mut taken = own.clone();
        let (mut after, mut depends) = (vec![false; n], vec![false; n]);
        (after[node.output], depends[node.output]) = (true, true);
        (after[value], depends[value]) = (true, true);
        let mut end = None;

        for (u, un) in nodes.iter().enumerate().skip(i + 1) {
            depends[un.output] = un.inputs.iter().any(|&v| depends[v]);

            if !live[u] || !un.inputs.iter().any(|&v| after[v]) || own.contains(&u) {
                continue;
            }

            let expands = pair.is_none() && (backend.expansion)(graph, node, un);

            let updates_slot = (un.inputs.get(1)).is_some_and(|&v| after[v] && update(un, v));

            let step = elementwise(&un.primitive)
                || matches!(un.primitive, Primitive::Reshape { .. })
                || (backend.swap_epilogues && matches!(un.primitive, Primitive::SwapPairs))
                || pair.is_some_and(|p| p.contains(&u))
                || expands
                || updates_slot;

            let fuses = (fusible[u] || updates_slot) && step && !in_row(u) && !claimed[u];
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
            // Keep only taken nodes the current end needs.
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

            // An auxiliary value only read by a narrowing cast is written already narrowed.
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

            // Paired GEMMs need both halves in the chain and every output half-sized.
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
                && aux.len() <= backend.max_aux
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

            updates.extend(
                taken
                    .iter()
                    .filter(|&&k| matches!(nodes[k].primitive, Primitive::DynamicUpdateSlice)),
            );

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
        updates,
    }
}

/// Whether node `i` is a split-K combine: a float sum over axis 0 (the K chunks, the dot's leading batch dim) of
/// a batched dot's products that only it reads.
fn combine(a: &Analysis, i: usize) -> bool {
    let (graph, node) = (a.graph, &a.nodes[i]);

    let Primitive::ReduceSum { axes, .. } = &node.primitive else {
        return false;
    };

    let products = node.inputs[0];

    let batched = a.producer[products].is_some_and(|d| {
        matches!(&a.nodes[d].primitive, Primitive::DotGeneral { lhs_batch, .. } if !lhs_batch.is_empty())
    });

    *axes == [0]
        && batched
        && a.users[products] == [i]
        && !a.is_output[products]
        && graph.type_of(node.output).dtype.is_float()
        && a.fusible[i]
}
