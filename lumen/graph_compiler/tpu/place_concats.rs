//! Marks concatenations whose inputs can be written directly into the concat's buffer.
use super::{LANES, TpuModule};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::CompilerPass;

/// Whether every input of `concat` can be produced in place in its output buffer.
///
/// Requires concatenation on the last axis, lane-aligned input widths, inputs used only once by the
/// concat and made by a dot, elementwise fusion, fill, scalar broadcast or elementwise op.
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

    // Producers whose kernels can write a strided slice of the shared buffer.
    let writes = |p: &Primitive| match p {
        Primitive::DotGeneral { .. } | Primitive::Full { .. } => true,
        Primitive::Fusion { name, .. } => name == "dot_general" || name == "elementwise",
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

            // Look through fusion outputs to the fusion that made them.
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

/// Sets `in_place` on every placeable concatenation so producers write into its buffer.
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
