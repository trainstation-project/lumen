//! Triton in-place concatenation: operands written straight into their slot of the result by the
//! kernels producing them, so the concatenate copies nothing.

use crate::codegen::triton::fusible;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Compiler pass marking concatenations whose operands' kernels can write their slots.
pub(super) struct PlaceConcats;

impl<M: Module> CompilerPass<M> for PlaceConcats {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "place_concats"
    }

    /// Marks the concatenations `in_place` accepts.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let concats = in_place(m.graph());

        for c in concats {
            if let Primitive::Concatenate { in_place, .. } = &mut m.graph_mut().nodes[c].primitive {
                *in_place = true;
            }
        }

        Ok(())
    }
}

/// Concatenations each of whose operands is read only there, once, is no program output and is
/// produced by a node that ends a kernel of its own writing fresh memory: a matmul, or a fusible
/// elementwise or copying layout op (a reshape or view slice aliases its operand instead). Loop
/// fusion leaves such a concatenate unfused, so each producer is its kernel's root and writes its
/// slot through the slot's strides.
fn in_place(graph: &Graph) -> Vec<usize> {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    let mut readers: Vec<usize> = vec![0; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| readers[v] += 1);
    }

    let writes = |v: usize| {
        let Some(p) = producer[v] else {
            return false;
        };

        let node = &nodes[p];

        let copies = matches!(
            node.primitive,
            Primitive::Transpose { .. }
                | Primitive::BroadcastInDim { .. }
                | Primitive::Slice { view: false, .. }
        );

        let kernel = fusible(graph, node) && (elementwise(&node.primitive) || copies);
        kernel || matches!(node.primitive, Primitive::DotGeneral { .. })
    };

    nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| matches!(node.primitive, Primitive::Concatenate { .. }))
        .filter(|(_, node)| {
            node.inputs
                .iter()
                .all(|&v| readers[v] == 1 && !graph.outputs().contains(&v) && writes(v))
        })
        .map(|(c, _)| c)
        .collect()
}
