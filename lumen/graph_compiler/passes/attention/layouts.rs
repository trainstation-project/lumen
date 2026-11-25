//! Adapting attention fusions (both directions) to kernels that read and write whole tiles (TPU's): the
//! `attention_layouts` pass, right after the attention passes. It merges the last two axes of outputs written with
//! rows narrower than their last axis, folds reshape, sum and cast chains that read attention outputs into the
//! fusion bodies (a decode attention's only when its kernel writes the result, `takes`), moves the backward's `delta` dot into the backward fusion (`delta_inside`), gives a fusion the
//! backend's kernels can't take its operands heads-major (`heads_major`), and checks the backend's attention kernels
//! take every attention fusion.
//!
//! Devices: it runs, with `flash_attention`, on each device implementing `TiledAttentions` (its attention-kernel
//! check, in `attention/<device>.rs`): TPU.

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, attention};

/// Indices of the attention fusion's outputs written with rows narrower than their last axis (a forward attention of one
/// query a head writes a matrix of its heads, its row stride no layout).
fn unreadable(body: &Graph, forward: bool) -> Vec<usize> {
    if forward && attention::of_body(body).is_some_and(|a| a.sq == 1) {
        return Vec::new();
    }

    let wide = |acc: &attention::Access| {
        let shape = &body.type_of(acc.base).shape;
        shape.len() >= 3 && acc.row != shape[shape.len() - 1]
    };

    let index = |acc: &attention::Access| body.outputs().iter().position(|&o| o == acc.base);

    let accesses = match forward {
        true => forward_outputs(body),
        false => backward_outputs(body),
    };

    accesses
        .iter()
        .filter(|acc| wide(acc))
        .filter_map(index)
        .collect()
}

/// Round-trips each such attention output through a reshape merging its last two axes.
///
/// Outputs already read by a reshape or sum are left for `reshaped_outputs` to fold instead.
fn merged_outputs(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }

    let laid_out = |v: Var| {
        readers[v].iter().any(|&r| {
            matches!(
                nodes[r].primitive,
                Primitive::Reshape { .. } | Primitive::ReduceSum { .. }
            )
        })
    };

    let mut merged = vec![false; graph.types.len()];

    for node in nodes {
        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            continue;
        };

        if !name.starts_with("flash_attention") {
            continue;
        }

        for k in unreadable(body, name == "flash_attention") {
            let v = match k {
                0 => Some(node.output),
                _ => readers[node.output]
                    .iter()
                    .find_map(|&u| match nodes[u].primitive {
                        Primitive::FusionOutput { index, .. } if index == k => {
                            Some(nodes[u].output)
                        }
                        _ => None,
                    }),
            };

            if let Some(v) = v.filter(|&v| !laid_out(v)) {
                merged[v] = true;
            }
        }
    }

    if !merged.contains(&true) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // `FusionOutput` nodes read the fusion itself, not the reshaped round trip.
    let mut own: Vec<Var> = vec![0; graph.types.len()];

    for node in nodes {
        out.set_origin(node);

        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| match node.primitive {
                Primitive::FusionOutput { .. } => own[v],
                _ => map[v],
            })
            .collect();

        let v = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");

        out.set_label(v, node.label);
        (map[node.output], own[node.output]) = (v, v);

        // Readers take the value through the merged shape, which matches how the kernel writes
        // these rows.
        if merged[node.output] {
            let shape = &graph.type_of(node.output).shape;
            let r = shape.len();
            let mut flat = shape[..r - 2].to_vec();
            flat.push(shape[r - 2] * shape[r - 1]);

            let m = out
                .apply(Primitive::Reshape { new_sizes: flat }, &[v])
                .expect("the output merged");

            map[node.output] = out
                .apply(
                    Primitive::Reshape {
                        new_sizes: shape.clone(),
                    },
                    &[m],
                )
                .expect("the output as traced");
        }
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Folds reshape, sum and cast chains reading attention outputs into the fusion bodies. A forward fusion of one query a
/// head writes its heads' matrix as laid out unless the backend's kernels (`takes`) write its chain's result instead.
fn reshaped_outputs(graph: &Graph, takes: impl Fn(&str, &Graph) -> bool) -> Graph {
    let nodes = graph.nodes();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    let mut parts: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        for &v in &node.inputs {
            match node.primitive {
                Primitive::FusionOutput { .. } => parts[v].push(i),
                _ => readers[v].push(i),
            }
        }
    }

    let attention = |p: &Primitive| matches!(p, Primitive::Fusion { name, .. } if name.starts_with("flash_attention"));
    let reshape = |r: usize| matches!(nodes[r].primitive, Primitive::Reshape { .. });
    let reduce = |r: usize| matches!(nodes[r].primitive, Primitive::ReduceSum { .. });
    let cast = |r: usize| matches!(nodes[r].primitive, Primitive::Cast { .. });

    // A value read by exactly one node and not a graph output.
    let alone = |v: Var| match readers[v].as_slice() {
        [r] if !graph.outputs().contains(&v) => Some(*r),
        _ => None,
    };

    // A sole reshape or sum reader starts the chain; otherwise take the first reshape among many
    // readers. Further sole sums, casts and reshapes after them extend it only when the output has
    // one reader.
    let chain = |v: Var| {
        let first = readers[v].iter().copied().find(|&r| reshape(r));

        let mut chain: Vec<usize> = match alone(v) {
            Some(r) if reduce(r) || reshape(r) => vec![r],
            _ => first.into_iter().collect(),
        };

        while let Some(&last) = chain.last() {
            match alone(nodes[last].output) {
                Some(r)
                    if (reduce(r) || cast(r) || (reshape(r) && (reduce(last) || cast(last))))
                        && alone(v).is_some() =>
                {
                    chain.push(r)
                }
                _ => break,
            }
        }

        chain
    };

    let mut by_fusion: Vec<Vec<(usize, Vec<usize>)>> = vec![Vec::new(); nodes.len()];
    let mut skipped = vec![false; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        if !attention(&node.primitive) {
            continue;
        }

        // Forward fusions fold only their primary output; one of one query a head only a chain its kernel writes (its
        // heads in one row), else none (it writes its heads' matrix).
        let forward =
            matches!(&node.primitive, Primitive::Fusion { name, .. } if name == "flash_attention");

        if let Primitive::Fusion { name, body, .. } = &node.primitive
            && forward
            && attention::of_body(body).is_some_and(|a| a.sq == 1)
        {
            let chain = chain(node.output);
            let mut folded = body.clone();
            let mut outputs = folded.outputs().to_vec();

            for &r in &chain {
                outputs[0] = folded
                    .apply(nodes[r].primitive.clone(), &[outputs[0]])
                    .expect("the output's layout");
            }

            folded.set_outputs(&outputs).expect("the body's values");

            if !chain.is_empty() && takes(name, &folded) {
                chain.iter().for_each(|&r| skipped[r] = true);
                by_fusion[i].push((0, chain));
            }

            continue;
        }

        let mut values = vec![(0, node.output)];

        for &u in parts[node.output].iter().filter(|_| !forward) {
            if let Primitive::FusionOutput { index, .. } = nodes[u].primitive {
                values.push((index, nodes[u].output));
            }
        }

        for (index, v) in values {
            let chain = chain(v);

            if !chain.is_empty() {
                chain.iter().for_each(|&r| skipped[r] = true);
                by_fusion[i].push((index, chain));
            }
        }
    }

    if by_fusion.iter().all(Vec::is_empty) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // The rebuilt fusion behind each original fusion value, for its `FusionOutput` readers.
    let mut fused_as: Vec<Option<Var>> = vec![None; graph.types.len()];

    // The chain folded onto output `index` of fusion node `f`, if any.
    let folded = |f: usize, index: usize| {
        by_fusion[f]
            .iter()
            .find(|(k, _)| *k == index)
            .map(|(_, c)| c)
    };

    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);

        if skipped[i] {
            continue;
        }

        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| match node.primitive {
                Primitive::FusionOutput { .. } => fused_as[v].unwrap_or(map[v]),
                _ => map[v],
            })
            .collect();

        // Where this node's value comes from: output `index` of fusion node `f`.
        let at = match node.primitive {
            Primitive::FusionOutput { index, .. } => {
                Some((graph_producer(graph, node.inputs[0]), index))
            }
            _ if attention(&node.primitive) => Some((i, 0)),
            _ => None,
        };

        let chain = at.and_then(|(f, index)| folded(f, index));

        // Append each folded chain to its output in the body; `FusionOutput`s take the chain's final type.
        let primitive = match &node.primitive {
            Primitive::Fusion { name, label, body } if !by_fusion[i].is_empty() => {
                let mut body = body.clone();
                let mut outputs = body.outputs().to_vec();

                for (index, chain) in &by_fusion[i] {
                    for &r in chain {
                        outputs[*index] = body
                            .apply(nodes[r].primitive.clone(), &[outputs[*index]])
                            .expect("the output's layout");
                    }
                }

                body.set_outputs(&outputs).expect("the body's values");

                Primitive::Fusion {
                    name: name.clone(),
                    label,
                    body,
                }
            }
            Primitive::FusionOutput { index, .. } if chain.is_some() => Primitive::FusionOutput {
                index: *index,
                ty: graph
                    .type_of(nodes[*chain.and_then(|c| c.last()).expect("a chain")].output)
                    .clone(),
            },
            p => p.clone(),
        };

        let fused = out.apply(primitive, &inputs).expect("a node of the graph");
        out.set_label(fused, node.label);
        map[node.output] = fused;
        fused_as[node.output] = Some(fused);

        if let Some(chain) = chain {
            map[nodes[*chain.last().expect("a chain")].output] = fused;

            // A lone reshape folded out of several readers: restore the traced shape for the rest.
            if chain.len() == 1 && reshape(chain[0]) {
                let back = Primitive::Reshape {
                    new_sizes: graph.type_of(node.output).shape.clone(),
                };

                map[node.output] = out.apply(back, &[fused]).expect("the output as traced");
            }
        }
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Index of the node producing `v`.
fn graph_producer(graph: &Graph, v: Var) -> usize {
    graph
        .nodes()
        .iter()
        .position(|n| n.output == v)
        .expect("a node's value")
}

/// Whether the backend's kernels (`D::check`) take the attention fusion `name` of `body` once `heads_major` has laid out
/// its inputs.
fn takes<D: TiledAttentions>(name: &str, body: &Graph) -> bool {
    let mut graph = Graph::new();

    let inputs: Vec<Var> = body
        .inputs()
        .iter()
        .map(|&v| graph.input(body.type_of(v).clone()))
        .collect();

    let fusion = Primitive::Fusion {
        name: name.into(),
        label: crate::graph::intern(name.into()),
        body: body.clone(),
    };

    let value = graph.apply(fusion, &inputs).expect("the fusion alone");
    graph.set_outputs(&[value]).expect("its value");

    heads_major::<D>(&graph)
        .nodes()
        .iter()
        .all(|n| match &n.primitive {
            Primitive::Fusion { name, body, .. } => D::check(name, body).is_ok(),
            _ => true,
        })
}

/// Checks that the backend's attention kernels (`D::check`) take every attention fusion left in `graph`.
fn attentions<D: TiledAttentions>(graph: &Graph) -> Result<(), String> {
    for node in graph.nodes() {
        if let Primitive::Fusion { name, body, .. } = &node.primitive
            && let "flash_attention" | "flash_attention_backward" = name.as_str()
        {
            D::check(name, body)?;
        }
    }

    Ok(())
}

/// A backend whose attention kernels read and write whole tiles (TPU's): the `attention_layouts`
/// pass adapts its attention fusions to them.
pub(crate) trait TiledAttentions {
    /// Whether the attention kernels take fusion `body` named `name`: the error when not.
    fn check(name: &str, body: &crate::graph::Graph) -> Result<(), String>;
}

/// Folds the flash-attention fusions' output layout chains and the backward's `delta` dot into the fusions, and
/// checks the TPU codegen takes each.
pub(crate) struct AttentionLayouts;

impl<D: TiledAttentions> CompilerPass<DeviceModule<'_, D>> for AttentionLayouts {
    /// With `flash_attention`.
    fn configured(config: &CompilerConfig) -> Option<Self> {
        (config.flash_attention).then_some(AttentionLayouts)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "attention_layouts"
    }

    /// Rewrites the graph's attention fusions for the backend's kernels and checks each.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        let reshaped = reshaped_outputs(&merged_outputs(&m.graph), takes::<D>);
        m.graph = heads_major::<D>(&delta_inside(&reshaped));
        attentions::<D>(&m.graph)
    }
}

/// How a forward fusion `body` writes its output.
fn forward_outputs(body: &Graph) -> Vec<attention::Access> {
    attention::of_body(body)
        .map(|a| a.out)
        .into_iter()
        .collect()
}

/// How a backward fusion `body` writes its dQ, dK and dV outputs.
fn backward_outputs(body: &Graph) -> Vec<attention::Access> {
    attention::backward_of_body(body)
        .map(|b| {
            [b.dq_out, b.dk_out, b.dv_out]
                .into_iter()
                .flatten()
                .map(|(_, acc)| acc)
                .collect()
        })
        .unwrap_or_default()
}

/// Moves the dot computing a backward fusion's `delta` input inside the fusion body.
///
/// The fusion then reads the dot's operands, through any reshape/transpose chains, instead of `delta`.
fn delta_inside(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let producer = |v: Var| nodes.iter().find(|n| n.output == v);

    // Walk back through reshapes and transposes to the value they were applied to.
    let unshaped = |mut v: Var| {
        let mut chain = Vec::new();

        while let Some(n) = producer(v).filter(|n| {
            matches!(
                n.primitive,
                Primitive::Reshape { .. } | Primitive::Transpose { .. }
            )
        }) {
            chain.push(n);
            v = n.inputs[0];
        }

        (v, chain)
    };

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    let mut changed = false;

    for node in nodes {
        out.set_origin(node);
        let mut inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        let mut primitive = node.primitive.clone();

        if let Primitive::Fusion { name, label, body } = &node.primitive
            && name == "flash_attention_backward"
            && let Some(b) = attention::backward_of_body(body)
            && let Some(k) = body.inputs().iter().position(|&v| v == b.delta.base)
            && let Some(dot) = producer(node.inputs[k])
            && matches!(dot.primitive, Primitive::DotGeneral { .. })
        {
            let operands: Vec<(Var, Vec<&Node>)> =
                dot.inputs.iter().map(|&v| unshaped(v)).collect();

            let mut inner = Graph::new();
            let mut within: Vec<Var> = vec![0; body.types.len()];
            let mut reads = Vec::new();
            let mut outer: Vec<(Var, Var)> = Vec::new();

            // Keep every body input except `delta`, which is now computed inside.
            for (j, &v) in body.inputs().iter().enumerate() {
                if j != k {
                    within[v] = inner.input(body.type_of(v).clone());
                    reads.push(map[node.inputs[j]]);
                    outer.push((node.inputs[j], within[v]));
                }
            }

            let mut sides = Vec::new();

            // Reuse an existing read when the dot's operand is already a fusion input.
            for (base, chain) in &operands {
                let base = *base;

                let mut x = match outer.iter().find(|(o, _)| *o == base) {
                    Some(&(_, x)) => x,
                    None => {
                        let x = inner.input(graph.type_of(base).clone());
                        reads.push(map[base]);
                        outer.push((base, x));
                        x
                    }
                };

                for r in chain.iter().rev() {
                    x = inner
                        .apply(r.primitive.clone(), &[x])
                        .expect("the operand's reshape or transpose");
                }

                sides.push(x);
            }

            within[body.inputs()[k]] = inner
                .apply(dot.primitive.clone(), &sides)
                .expect("delta's dot");

            for n in body.nodes() {
                let ins: Vec<Var> = n.inputs.iter().map(|&v| within[v]).collect();

                within[n.output] = inner
                    .apply(n.primitive.clone(), &ins)
                    .expect("a node of the body");
            }

            let outputs: Vec<Var> = body.outputs().iter().map(|&v| within[v]).collect();
            inner.set_outputs(&outputs).expect("the body's values");

            primitive = Primitive::Fusion {
                name: name.clone(),
                label,
                body: inner,
            };

            inputs = reads;
            changed = true;
        }
        map[node.output] = out.apply(primitive, &inputs).expect("a node of the graph");

        out.set_label(map[node.output], node.label);
    }

    if !changed {
        return graph.clone();
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// How operand `shape`, read through `acc` with `batch` extents, `seq` rows and `width` columns a head, is made
/// heads-major: the shape its buffer factors into by the access's strides (largest first), and the permutation
/// putting those factors in batch, row, column order; `None` when the strides don't factor the buffer (an offset, a
/// gap, a repeat) or it already is heads-major, its buffer's last two dims its last two factors.
fn heads_layout(
    shape: &[usize],
    acc: &attention::Access,
    batch: &[usize],
    (seq, width): (usize, usize),
) -> Option<(Vec<usize>, Vec<usize>)> {
    if acc.offset != 0 || acc.batch.len() != batch.len() {
        return None;
    }

    // Each factor's (stride, size, order in heads-major), dropping those of one element.
    let mut factors: Vec<(usize, usize, usize)> = acc
        .batch
        .iter()
        .zip(batch)
        .enumerate()
        .map(|(j, (&(stride, div), &n))| (stride, n / div, j))
        .chain([
            (acc.row, seq, batch.len()),
            (acc.col, width, batch.len() + 1),
        ])
        .filter(|&(_, size, _)| size > 1)
        .collect();

    factors.sort_by_key(|f| std::cmp::Reverse(f.0));

    // The factors must tile the buffer exactly, the last of stride 1.
    let mut expected = 1;

    for &(stride, size, _) in factors.iter().rev() {
        if stride != expected {
            return None;
        }

        expected *= size;
    }

    if expected != shape.iter().product::<usize>() {
        return None;
    }

    let sizes: Vec<usize> = factors.iter().map(|f| f.1).collect();
    let mut perm: Vec<usize> = (0..factors.len()).collect();
    perm.sort_by_key(|&d| factors[d].2);

    // Moved when the factors aren't heads-major, or are but the buffer's last two dims (the tiles the kernels read)
    // aren't its last two factors (a decode query read from a projection's flat row).
    let tail = |dims: &[usize]| dims[dims.len().saturating_sub(2)..].to_vec();
    let moves = perm.iter().enumerate().any(|(d, &p)| d != p) || tail(shape) != tail(&sizes);

    moves.then_some((sizes, perm))
}

/// An operand's factors (its buffer's shape by the access's strides) and the permutation making them heads-major.
type Layout = (Vec<usize>, Vec<usize>);

/// Gives each attention fusion the backend's kernels (`D::check`) can't take its q, k, v (and the backward's dO)
/// heads-major: a reshape and transpose before the fusion write each such body input heads-major (`heads_layout`),
/// and the body transposes and reshapes it back, so the kernel reads each head as one contiguous slab. A forward
/// attention's output stays written through its layout chain when the kernels take that once the inputs are moved;
/// else the chain runs after the fusion.
fn heads_major<D: TiledAttentions>(graph: &Graph) -> Graph {
    let nodes = graph.nodes();

    // Per attention fusion node: its body inputs to lay out heads-major, each's factors and their permutation, and
    // the body nodes of its first output's trailing layout chain to run after it instead.
    let mut moved: Vec<Vec<(usize, Layout)>> = vec![Vec::new(); nodes.len()];
    let mut trailing: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        let Primitive::Fusion { name, body, .. } = &node.primitive else {
            continue;
        };

        // Each operand's access, with the batch extents and its rows and columns a head.
        let accesses = match name.as_str() {
            "flash_attention" => attention::of_body(body).map(|a| {
                let at = |acc, seq, width| (acc, a.batch.clone(), (seq, width));
                vec![at(a.q, a.sq, a.h), at(a.k, a.sk, a.h), at(a.v, a.sk, a.hv)]
            }),
            "flash_attention_backward" => attention::backward_of_body(body).map(|b| {
                let at = |acc, seq, width| (acc, b.batch.clone(), (seq, width));
                vec![
                    at(b.q, b.sq, b.h),
                    at(b.k, b.sk, b.h),
                    at(b.v, b.sk, b.hv),
                    at(b.d_o, b.sq, b.hv),
                ]
            }),
            _ => None,
        };

        let Some(accesses) = accesses.filter(|_| D::check(name, body).is_err()) else {
            continue;
        };

        for (acc, batch, extents) in &accesses {
            let Some(k) = body.inputs().iter().position(|&v| v == acc.base) else {
                continue;
            };

            let shape = &body.type_of(acc.base).shape;
            let layout = heads_layout(shape, acc, batch, *extents);

            if let Some(layout) = layout.filter(|_| moved[i].iter().all(|(j, _)| *j != k)) {
                moved[i].push((k, layout));
            }
        }

        // The output's layout chain runs after the fusion only if the kernel can't write through it once the inputs
        // are heads-major; one query a head writes its heads' matrix as laid out (`unreadable`): its chain stays.
        let inputs_only = transposed_back(body, &moved[i], &[]);

        if name == "flash_attention"
            && attention::of_body(body).is_some_and(|a| a.sq > 1)
            && D::check(name, &inputs_only).is_err()
        {
            trailing[i] = layout_chain(body);
        }
    }

    if moved.iter().all(Vec::is_empty) && trailing.iter().all(Vec::is_empty) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // `FusionOutput` nodes read the fusion itself, not its first output's layout chain.
    let mut own: Vec<Var> = vec![0; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);

        let mut inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| match node.primitive {
                Primitive::FusionOutput { .. } => own[v],
                _ => map[v],
            })
            .collect();

        let primitive = match &node.primitive {
            Primitive::Fusion { name, label, body }
                if !moved[i].is_empty() || !trailing[i].is_empty() =>
            {
                for (k, (sizes, perm)) in &moved[i] {
                    let factored = out
                        .apply(
                            Primitive::Reshape {
                                new_sizes: sizes.clone(),
                            },
                            &[inputs[*k]],
                        )
                        .expect("the operand's factors");

                    inputs[*k] = out
                        .apply(
                            Primitive::Transpose {
                                permutation: perm.clone(),
                            },
                            &[factored],
                        )
                        .expect("the operand heads-major");
                }

                Primitive::Fusion {
                    name: name.clone(),
                    label,
                    body: transposed_back(body, &moved[i], &trailing[i]),
                }
            }
            p => p.clone(),
        };

        let v = out.apply(primitive, &inputs).expect("a node of the graph");
        out.set_label(v, node.label);
        (map[node.output], own[node.output]) = (v, v);

        // The first output's layout chain, now after the fusion.
        if let Primitive::Fusion { body, .. } = &node.primitive {
            for &n in &trailing[i] {
                map[node.output] = out
                    .apply(body.nodes()[n].primitive.clone(), &[map[node.output]])
                    .expect("the output's layout");
            }
        }
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// The body nodes of the transposes and reshapes ending in `body`'s first output, in order, when they hold a transpose
/// and no other node or output reads their values: the attention written before them is heads-major.
fn layout_chain(body: &Graph) -> Vec<usize> {
    let nodes = body.nodes();
    let mut chain = Vec::new();
    let mut v = body.outputs()[0];

    let read = |u: Var| nodes.iter().filter(|n| n.inputs.contains(&u)).count();

    while let Some(n) = nodes.iter().position(|n| n.output == v) {
        let layout = matches!(
            nodes[n].primitive,
            Primitive::Transpose { .. } | Primitive::Reshape { .. }
        );

        let alone = chain.is_empty() || (read(v) == 1 && !body.outputs().contains(&v));

        if !layout || !alone {
            break;
        }

        chain.push(n);
        v = nodes[n].inputs[0];
    }

    chain.reverse();

    match chain
        .iter()
        .any(|&n| matches!(nodes[n].primitive, Primitive::Transpose { .. }))
    {
        true => chain,
        false => Vec::new(),
    }
}

/// `body` with each input `k` of `moved` taken heads-major (its factors permuted) and turned back into its shape, and
/// its first output the value before its `trailing` layout chain, which no longer runs in it.
fn transposed_back(body: &Graph, moved: &[(usize, Layout)], trailing: &[usize]) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; body.types.len()];

    for (k, &v) in body.inputs().iter().enumerate() {
        let ty = body.type_of(v);

        map[v] = match moved.iter().find(|(j, _)| *j == k) {
            Some((_, (sizes, perm))) => {
                let shape: Vec<usize> = perm.iter().map(|&d| sizes[d]).collect();
                out.input(crate::graph::TensorType::new(ty.dtype, &shape))
            }
            None => out.input(ty.clone()),
        };
    }

    // Each moved input transposed back to its factors, after every input; a reshape reads the factors (a strided view
    // the attention's accesses follow), any other node the factors reshaped to the input's shape.
    let mut factors: Vec<Option<Var>> = vec![None; body.types.len()];

    for (k, (_, perm)) in moved {
        let v = body.inputs()[*k];
        let mut inverse = vec![0; perm.len()];
        perm.iter().enumerate().for_each(|(d, &p)| inverse[p] = d);

        let factored = out
            .apply(
                Primitive::Transpose {
                    permutation: inverse,
                },
                &[map[v]],
            )
            .expect("the operand's factors");

        factors[v] = Some(factored);

        let other = body
            .nodes()
            .iter()
            .any(|n| n.inputs.contains(&v) && !matches!(n.primitive, Primitive::Reshape { .. }));

        map[v] = match other || body.outputs().contains(&v) {
            true => out
                .apply(
                    Primitive::Reshape {
                        new_sizes: body.type_of(v).shape.clone(),
                    },
                    &[factored],
                )
                .expect("the operand as traced"),
            false => factored,
        };
    }

    for (n, node) in body.nodes().iter().enumerate() {
        if trailing.contains(&n) {
            continue;
        }

        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| match (&node.primitive, factors[v]) {
                (Primitive::Reshape { .. }, Some(f)) => f,
                _ => map[v],
            })
            .collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the body");
    }

    let mut outputs: Vec<Var> = body.outputs().iter().map(|&v| map[v]).collect();

    if let Some(&first) = trailing.first() {
        outputs[0] = map[body.nodes()[first].inputs[0]];
    }

    out.set_outputs(&outputs).expect("values of the body");
    out
}
