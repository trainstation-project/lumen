//! In-place concatenation: the kernels producing a `Concatenate`'s operands write them straight into their slot of the
//! result, so the concatenation copies nothing. Two slots use it: `place_concats` marks concatenations
//! (`in_place = true`) before `fusion`, right after the dot passes; `concats` runs after `fusion`, `place_views` and
//! `alone` and unmarks the ones fusion made invalid, each then copied by a kernel of its own.
//!
//! Devices: each device implementing `Slots` (what its kernels can write through a slot's strides, in `<device>.rs`)
//! runs `place_concats`: MPS and Triton with `fuse`, TPU always. `concats` runs on those implementing `Rechecks` (the
//! slot rule after fusion): MPS and TPU. CPU and CUDA run neither.
//!
//! The rule: `in_place_concats(graph, writes, worth)` returns the live concatenations where every operand is read only
//! by that concatenation (ignoring `FusionOutput` readers), exactly once, is no program output, and is made by a
//! kernel `writes(concat, k, producer, index)` accepts (for an operand that is output `index` of a fusion, the producer
//! is that fusion), and at least one operand is `worth` placing. `Slots::worth` defaults to `worth`, which MPS and
//! Triton share: an operand is worth placing when a kernel of its own makes it (a matmul, attention's backward, a
//! fusion), whose output written into its slot needs no copy. TPU places whenever it can.
//!
//! `passes/fusion` treats in-place concatenations as unfusible, and `dot/fuse/tpu.rs` calls `in_place_concats` to avoid
//! building a placed concat inside a dot fusion.

#[cfg(lumen_mps_linked)]
mod mps;
pub(crate) mod tpu;
mod triton;

use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::passes::CompilerPass;
use crate::graph_compiler::passes::alone::wrapped;

/// Whether an operand `producer` makes is worth placing its concatenation for: when `producer` is a kernel of its
/// own (a matmul, attention's backward, a fusion), whose output written into its slot needs no copy. A cheap
/// elementwise or layout producer is not: fusion builds the concatenation inside a kernel that runs anyway, its
/// readers' (rotary's `cat([-x2, x1])` before `* sin`) or its other operands'.
pub(crate) fn worth(producer: &Node) -> bool {
    let layout = matches!(
        producer.primitive,
        Primitive::Reshape { .. }
            | Primitive::Transpose { .. }
            | Primitive::BroadcastInDim { .. }
            | Primitive::Slice { .. }
    );

    !(layout || crate::graph_compiler::elementwise(&producer.primitive))
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

/// A module whose backend's kernels may write a concatenation's operands into their slots, which the `place_concats`
/// pass marks in place. Operand `k` of
/// `concat` is made by `producer`, the fusion itself when the operand is its output `index`.
pub(crate) trait Slots: crate::graph_compiler::passes::Module {
    /// Whether the kernel making operand `k` of `concat` can write it into its slot.
    fn writes(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool;

    /// Whether writing operand `k` into its slot is worth placing `concat` for; a concatenation is placed
    /// when some operand is. By default, `worth`'s rule (MPS's and Triton's).
    fn worth(
        &self,
        _concat: &crate::graph::Node,
        _k: usize,
        producer: &crate::graph::Node,
        _index: usize,
    ) -> bool {
        crate::graph_compiler::passes::place_concats::worth(producer)
    }
}

/// A module whose fusion may leave a concatenation it was to write in place unable to be (the graph compiler's
/// `concats` pass, after fusion): `written` is the slot rule then; each concatenation unmarked gets a kernel of its own,
/// named as `Lone::kernel` names one-node fusions.
pub(crate) trait Rechecks: Slots + crate::graph_compiler::passes::alone::Lone {
    /// Whether, after fusion, the kernel making operand `k` of `concat` writes it into its slot.
    fn written(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        index: usize,
    ) -> bool;
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

/// Compiler pass copying the concatenations fusion left unable to be written in place.
pub(crate) struct Concats;

impl<M: Rechecks> CompilerPass<M> for Concats {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "concats"
    }

    /// Unmarks in-place concatenations that fusion invalidated (`Rechecks::written`); each gets a kernel of its own
    /// copying it (named as `alone` names its kernels).
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let valid = in_place_concats(
            m.graph(),
            |c, k, p, index| m.written(c, k, p, index),
            |_, _, p, _| worth(p),
        );

        let mut unmarked = false;

        for (i, node) in m.graph_mut().nodes.iter_mut().enumerate() {
            if let Primitive::Concatenate { in_place, .. } = &mut node.primitive
                && *in_place
                && !valid.contains(&i)
            {
                *in_place = false;
                unmarked = true;
            }
        }

        if unmarked {
            let copying = |n: &Node| {
                matches!(
                    n.primitive,
                    Primitive::Concatenate {
                        in_place: false,
                        ..
                    }
                )
            };

            let graph = m.graph().clone();
            *m.graph_mut() = wrapped(&graph, &[], copying, |body, by_value| {
                m.kernel(body, by_value)
            });
        }

        Ok(())
    }
}
