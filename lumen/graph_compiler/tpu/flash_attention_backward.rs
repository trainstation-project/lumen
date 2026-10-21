use super::TpuModule;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::attention;
use crate::graph_compiler::pass::CompilerPass;

fn delta_inside(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let producer = |v: Var| nodes.iter().find(|n| n.output == v);
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
            for (j, &v) in body.inputs().iter().enumerate() {
                if j != k {
                    within[v] = inner.input(body.type_of(v).clone());
                    reads.push(map[node.inputs[j]]);
                    outer.push((node.inputs[j], within[v]));
                }
            }
            let mut sides = Vec::new();
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

fn unreadable(body: &Graph, forward: bool) -> Vec<usize> {
    let wide = |acc: &attention::Access| {
        let shape = &body.type_of(acc.base).shape;
        shape.len() >= 3 && acc.row != shape[shape.len() - 1]
    };
    let index = |acc: &attention::Access| body.outputs().iter().position(|&o| o == acc.base);
    let accesses: Vec<attention::Access> = match forward {
        true => attention::of_body(body)
            .map(|a| a.out)
            .into_iter()
            .collect(),
        false => attention::backward_of_body(body)
            .map(|b| {
                [b.dq_out, b.dk_out, b.dv_out]
                    .into_iter()
                    .flatten()
                    .map(|(_, acc)| acc)
                    .collect()
            })
            .unwrap_or_default(),
    };
    accesses
        .iter()
        .filter(|acc| wide(acc))
        .filter_map(index)
        .collect()
}

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
    let alone = |v: Var| match readers[v].as_slice() {
        [r] if !graph.outputs().contains(&v) => Some(*r),
        _ => None,
    };
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
    let mut fused_as: Vec<Option<Var>> = vec![None; graph.types.len()];
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
        let at = match node.primitive {
            Primitive::FusionOutput { index, .. } => {
                Some((graph_producer(graph, node.inputs[0]), index))
            }
            _ if attention(&node.primitive) => Some((i, 0)),
            _ => None,
        };
        let chain = at.and_then(|(f, index)| folded(f, index));
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

fn graph_producer(graph: &Graph, v: Var) -> usize {
    graph
        .nodes()
        .iter()
        .position(|n| n.output == v)
        .expect("a node's value")
}

fn attentions(graph: &Graph) -> Result<(), String> {
    for node in graph.nodes() {
        if let Primitive::Fusion { name, body, .. } = &node.primitive {
            match name.as_str() {
                "flash_attention" => crate::ops::attention::tpu::forward(body).map(drop)?,
                "flash_attention_backward" => {
                    crate::ops::attention::tpu::backward(body).map(drop)?
                }
                _ => {}
            }
        }
    }
    Ok(())
}

pub(super) struct FlashAttentionBackward;

impl CompilerPass<TpuModule<'_>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph =
            attention::fuse_backward(&m.graph, |_| false, |_| "flash_attention_backward".into());
        m.graph = delta_inside(&reshaped_outputs(&merged_outputs(&m.graph)));
        attentions(&m.graph)
    }
}
