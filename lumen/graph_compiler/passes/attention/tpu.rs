//! How TPU adapts attention fusions (both directions) to its kernels' layouts: folds output layout chains and
//! the backward's `delta` dot (`backward::tpu`) into the fusions, then checks the TPU codegen takes each.

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::attention;
use crate::graph_compiler::tpu::TpuModule;

/// Indices of the attention fusion's outputs written with rows narrower than their last axis.
fn unreadable(body: &Graph, forward: bool) -> Vec<usize> {
    let wide = |acc: &attention::Access| {
        let shape = &body.type_of(acc.base).shape;
        shape.len() >= 3 && acc.row != shape[shape.len() - 1]
    };

    let index = |acc: &attention::Access| body.outputs().iter().position(|&o| o == acc.base);

    let accesses = match forward {
        true => super::forward::tpu::outputs(body),
        false => super::backward::tpu::outputs(body),
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

/// Folds reshape, sum and cast chains reading attention outputs into the fusion bodies.
fn reshaped_outputs(graph: &Graph) -> Graph {
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

        // Forward fusions fold only their primary output.
        let forward =
            matches!(&node.primitive, Primitive::Fusion { name, .. } if name == "flash_attention");

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

/// Checks that the TPU codegen supports every attention fusion left in `graph`.
fn attentions(graph: &Graph) -> Result<(), String> {
    for node in graph.nodes() {
        if let Primitive::Fusion { name, body, .. } = &node.primitive
            && let "flash_attention" | "flash_attention_backward" = name.as_str()
        {
            crate::codegen::attention::tpu::kernel(name, body)?;
        }
    }

    Ok(())
}

/// Folds the flash-attention fusions' output layout chains and the backward's `delta` dot into the fusions, and
/// checks the TPU codegen takes each.
pub(crate) struct AttentionLayouts;

impl CompilerPass<TpuModule<'_>> for AttentionLayouts {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "attention_layouts"
    }

    /// Rewrites the graph's attention fusions for the TPU kernels and checks each.
    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = super::backward::tpu::delta_inside(&reshaped_outputs(&merged_outputs(&m.graph)));
        attentions(&m.graph)
    }
}
