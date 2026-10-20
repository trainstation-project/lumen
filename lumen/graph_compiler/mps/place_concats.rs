use std::collections::HashMap;

use super::{Kernel, MpsModule, PENDING, codegen, fusion};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct PlaceConcats;

impl<'a> CompilerPass<MpsModule<'a>> for PlaceConcats {
    fn name(&self) -> &'static str {
        "place_concats"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        for c in in_place_concats(&m.graph, &m.kernels, true) {
            if let Primitive::Concatenate { in_place, .. } = &mut m.graph.nodes[c].primitive {
                *in_place = true;
            }
        }
        Ok(())
    }
}

pub(super) fn in_place_concats(
    graph: &Graph,
    kernels: &[Kernel],
    before_fusion: bool,
) -> Vec<usize> {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut live = vec![false; graph.types.len()];
    graph.outputs().iter().for_each(|&v| live[v] = true);
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate().rev() {
        if live[node.output] {
            for &v in &node.inputs {
                live[v] = true;
                readers[v].push(i);
            }
        }
    }
    let kernel = |node: &Node| match &node.primitive {
        Primitive::Fusion { name, .. } => name
            .strip_prefix(PENDING)
            .and_then(|k| k.parse::<usize>().ok())
            .map(|k| &kernels[k]),
        _ => None,
    };
    let mut concats = Vec::new();
    for (c, node) in nodes.iter().enumerate() {
        if !live[node.output] || !matches!(node.primitive, Primitive::Concatenate { .. }) {
            continue;
        }
        let writes = node
            .inputs
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                let other =
                    |r: &&usize| !matches!(nodes[**r].primitive, Primitive::FusionOutput { .. });
                let alone =
                    readers[v].iter().filter(other).eq([&c]) && !graph.outputs().contains(&v);
                let once = node.inputs.iter().filter(|&&u| u == v).count() == 1;
                let p = producer[v].filter(|_| alone && once)?;
                let (f, index) = match nodes[p].primitive {
                    Primitive::FusionOutput { index, .. } => (producer[nodes[p].inputs[0]]?, index),
                    _ => (p, 0),
                };
                let fused = &nodes[f];
                match (kernel(fused), &fused.primitive) {
                    (Some(Kernel::AttentionBackward), Primitive::Fusion { body, .. }) => {
                        let atomic = body.outputs().len() == 3 && index == 2;
                        let at = graph.slot(node, k);
                        let views = HashMap::from([(index, (at.offset, at.strides))]);
                        let writes =
                            !atomic && attention::backward_of_body_into(body, views).is_some();
                        writes.then_some(true)
                    }
                    (Some(Kernel::Loop(_)), Primitive::Fusion { body, .. }) => {
                        codegen::writes_views(body).then_some(false)
                    }
                    (None, p) if before_fusion => {
                        let layout = matches!(
                            p,
                            Primitive::Reshape { .. }
                                | Primitive::Transpose { .. }
                                | Primitive::BroadcastInDim { .. }
                                | Primitive::Slice { .. }
                        );
                        let fuses =
                            fusion::fusible(graph, fused) && (layout || fusion::elementwise(p));
                        fuses.then_some(false)
                    }
                    _ => None,
                }
            })
            .collect::<Option<Vec<bool>>>();
        if writes.is_some_and(|w| w.contains(&true)) {
            concats.push(c);
        }
    }
    concats
}
