use super::{LANES, TpuModule};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::CompilerPass;

pub(super) fn placeable(
    graph: &Graph,
    producer: &[Option<usize>],
    users: &[Vec<usize>],
    concat: &Node,
) -> bool {
    let nodes = graph.nodes();
    let Primitive::Concatenate { dimension, .. } = concat.primitive else {
        return false;
    };
    let writes = |p: &Primitive| match p {
        Primitive::DotGeneral { .. } | Primitive::Full { .. } => true,
        Primitive::Fusion { name, .. } => name == "dot_general" || name == "loop",
        Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } => broadcast_dimensions.is_empty(),
        p => elementwise(p),
    };
    dimension + 1 == graph.type_of(concat.output).shape.len()
        && concat.inputs.iter().all(|&v| {
            let others = users[v]
                .iter()
                .filter(|&&u| !matches!(nodes[u].primitive, Primitive::FusionOutput { .. }));
            let alone = others.count() == 1 && !graph.outputs().contains(&v);
            let once = concat.inputs.iter().filter(|&&u| u == v).count() == 1;
            let made = producer[v].map(|p| match nodes[p].primitive {
                Primitive::FusionOutput { .. } => producer[nodes[p].inputs[0]],
                _ => Some(p),
            });
            alone
                && once
                && graph.type_of(v).shape[dimension].is_multiple_of(LANES)
                && made.flatten().is_some_and(|p| writes(&nodes[p].primitive))
        })
}

pub(super) struct PlaceConcats;

impl CompilerPass<TpuModule<'_>> for PlaceConcats {
    fn name(&self) -> &'static str {
        "place_concats"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        let nodes = m.graph.nodes();
        let mut producer: Vec<Option<usize>> = vec![None; m.graph.types.len()];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); m.graph.types.len()];
        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
            node.inputs.iter().for_each(|&v| users[v].push(i));
        }
        let placed: Vec<usize> = (0..nodes.len())
            .filter(|&c| placeable(&m.graph, &producer, &users, &nodes[c]))
            .collect();
        for c in placed {
            if let Primitive::Concatenate { in_place, .. } = &mut m.graph.nodes[c].primitive {
                *in_place = true;
            }
        }
        Ok(())
    }
}
