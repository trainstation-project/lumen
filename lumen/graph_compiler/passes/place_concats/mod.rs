//! In-place concatenation: operands written straight into their slot of the result by the kernels producing
//! them, so the concatenate copies nothing. `mps` holds MPS's slot rules and its pass, after fusion, copying
//! the concatenations fusion left unable to be written in place.

mod tpu;
mod triton;

use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Cuda, DeviceModule};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::graph_compiler::pipeline::Pass;

#[cfg(lumen_mps_linked)]
pub(crate) mod mps;

/// A module whose backend's kernels may write a concatenation's operands into their slots. Operand `k` of
/// `concat` is made by `producer`, the fusion itself when the operand is its output `index`.
pub(crate) trait Slots: Module {
    /// Whether the kernel making operand `k` of `concat` can write it into its slot.
    fn writes(&self, concat: &Node, k: usize, producer: &Node, index: usize) -> bool;

    /// Whether writing operand `k` into its slot is worth placing `concat` for; a concatenation is placed
    /// when some operand is. Any is, by default.
    fn worth(&self, _concat: &Node, _k: usize, _producer: &Node, _index: usize) -> bool {
        true
    }
}

/// Live concatenations each of whose operands is read only there, once, is no program output and is made
/// by a kernel `writes` accepts for its slot, with at least one operand `worth` placing.
pub(crate) fn in_place_concats(
    graph: &Graph,
    writes: impl Fn(&Node, usize, &Node, usize) -> bool,
    worth: impl Fn(&Node, usize, &Node, usize) -> bool,
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

    let mut concats = Vec::new();

    for (c, node) in nodes.iter().enumerate() {
        if !live[node.output] || !matches!(node.primitive, Primitive::Concatenate { .. }) {
            continue;
        }

        let placed = node
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

                // An operand a fusion makes as its output `index` is that fusion's to write.
                let (f, index) = match nodes[p].primitive {
                    Primitive::FusionOutput { index, .. } => (producer[nodes[p].inputs[0]]?, index),
                    _ => (p, 0),
                };

                writes(node, k, &nodes[f], index).then(|| worth(node, k, &nodes[f], index))
            })
            .collect::<Option<Vec<bool>>>();

        if placed.is_some_and(|worth| worth.contains(&true)) {
            concats.push(c);
        }
    }

    concats
}

/// Compiler pass marking the concatenations `in_place_concats` accepts with the module's `Slots::writes`.
pub(crate) struct PlaceConcats;

impl<M: Slots> CompilerPass<M> for PlaceConcats {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "place_concats"
    }

    /// Sets `in_place` on each concatenation whose operands' kernels can write their slots.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let concats = in_place_concats(
            m.graph(),
            |c, k, p, index| m.writes(c, k, p, index),
            |c, k, p, index| m.worth(c, k, p, index),
        );

        for c in concats {
            if let Primitive::Concatenate { in_place, .. } = &mut m.graph_mut().nodes[c].primitive {
                *in_place = true;
            }
        }

        Ok(())
    }
}

/// What each device runs for this pass (each device's in its own file here); a device it doesn't run on runs the
/// identity pass.
pub(crate) trait PlaceConcatsDevice: Sized + 'static {
    /// Places concatenations whose operands can be written straight into their slots.
    fn place_concats<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }

    /// Copies the concatenations fusion left unable to be written in place.
    fn concats<'a>(_: &CompilerConfig) -> Pass<DeviceModule<'a, Self>> {
        Box::new(Identity)
    }
}

/// Devices this pass doesn't run on: the identity pass.
impl PlaceConcatsDevice for Cpu {}
impl PlaceConcatsDevice for Cuda {}
