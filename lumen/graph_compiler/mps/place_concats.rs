//! The concatenates computed in place: each operand written at its place
//! in the result by the kernel computing it, marked so in the graph
//! ([`Primitive::Concatenate`]'s `in_place`, [`Graph::placements`]).

use std::collections::HashMap;

use super::{Kernel, MpsModule, PENDING, codegen, fusion};
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::attention;
use crate::graph_compiler::pass::CompilerPass;

/// The concatenates whose operands' kernels will write them in place
/// ([`in_place_concats`], before loop fusion makes them): marked so
/// ([`Primitive::Concatenate`]'s `in_place`), which every pass after
/// respects and the plan reads ([`Graph::placements`]).
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

/// The concatenates of `graph` that can be computed in place (by node,
/// [`PlaceConcats`] marks them): concatenates of an attention backward's gradients
/// (`[dq | dk | dv]`, which `merge_dots_backward` merges attention's
/// projections' gradients' dots on), each operand read by the concatenate
/// alone and written at its place in the result ([`Graph::slot`]) by its kernel:
/// an attention backward's, through its layout chain
/// ([`attention::fuse_backward`]; not a dQ it adds atomically, into memory
/// zeroed whole first), or a loop fusion's that writes views
/// ([`codegen::writes_views`]: that dQ cast and laid out). No kernel
/// copies them side by side.
///
/// `before_fusion`: those [`fusion`] leaves be, its loop fusions not made
/// yet: an operand it would compute in one (an elementwise or layout
/// primitive's value) counts as one. One whose fusion then cannot write
/// it there is concatenated by a kernel of its own ([`Codegen`]).
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
    // The live nodes reading each value.
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
        // Whether each operand's kernel writes it at its place (one fusion
        // will make counting), and whether it is an attention backward's.
        let writes = node
            .inputs
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                // (A fusion's other outputs read its value: not a reader.)
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
