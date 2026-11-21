//! Pass that wraps the nodes left outside any fusion that only a generated kernel computes into
//! single-node `Fusion` kernels. It runs near the end of the device pipeline, after `fusion` and `place_views`,
//! so codegen compiles every remaining kernel the same way; it is always on, with no config flag.
//!
//! Devices: it runs on each device whose codegen implements `codegen::Lone` (which lone nodes only a generated
//! kernel computes, which inputs kernels take by value, each kernel's name): CUDA (`codegen/devices/cuda.rs`), MPS
//! (`codegen/devices/mps`) and Triton (`codegen/devices/triton`). TPU and CPU get the identity pass.
//!
//! The rewrite:
//! - Each node the backend's `Lone::alone` accepts becomes a `Fusion` whose body is that node alone; an operand
//!   it reads twice becomes a single fusion input. The fusion keeps the node's label; `Lone::kernel` names it.
//! - By-value scalar inputs stay by value only when every reader is a fusion or becomes one and the input is no
//!   graph output; otherwise they are dropped from the list, since an unfused reader needs them as buffers.
//! - `wrapped` is the rewrite on its own; other code calls it with any predicate.

use crate::codegen::Lone;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::passes::CompilerPass;

/// Compiler pass wrapping every lone node the backend has no other kernel for in its own one-node fusion.
pub(crate) struct Alone;

impl<M: Lone> CompilerPass<M> for Alone {
    /// Always.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(Alone)
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "alone"
    }

    /// Drops by-value scalars that some unfused consumer still reads, then wraps the lone nodes.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        let graph = std::mem::replace(m.graph_mut(), Graph::new());

        // A scalar input can stay by-value only if every reader becomes a fusion kernel that takes it as a constant.
        let by_value = |i: &usize| {
            let v = graph.inputs()[*i];

            let fused =
                |n: &Node| matches!(n.primitive, Primitive::Fusion { .. }) || M::alone(&graph, n);

            !graph.outputs().contains(&v)
                && graph
                    .nodes()
                    .iter()
                    .filter(|n| n.inputs.contains(&v))
                    .all(fused)
        };

        let scalars: Vec<Var> = match m.scalars() {
            Some(scalars) => {
                scalars.retain(by_value);
                scalars.iter().map(|&i| graph.inputs()[i]).collect()
            }
            None => Vec::new(),
        };

        *m.graph_mut() = wrapped(
            &graph,
            &scalars,
            |n| M::alone(&graph, n),
            |body, by_value| m.kernel(body, by_value),
        );

        Ok(())
    }
}

/// Rebuilds `graph`, replacing each node `wraps` accepts by a `Fusion` whose body is that node alone.
/// `kernel` gets the body and one flag per distinct input saying whether it is in `scalars`, and
/// returns the kernel name.
pub(crate) fn wrapped(
    graph: &Graph,
    scalars: &[Var],
    wraps: impl Fn(&Node) -> bool,
    mut kernel: impl FnMut(&Graph, &[bool]) -> String,
) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    for node in graph.nodes() {
        out.set_origin(node);

        if wraps(node) {
            // Deduplicate operands so a value read twice becomes a single fusion input.
            let mut reads: Vec<Var> = Vec::new();

            for &v in &node.inputs {
                if !reads.contains(&v) {
                    reads.push(v);
                }
            }

            let mut body = Graph::new();

            let ins: Vec<Var> = reads
                .iter()
                .map(|&v| body.input(graph.type_of(v).clone()))
                .collect();

            let operands: Vec<Var> = node
                .inputs
                .iter()
                .map(|v| ins[reads.iter().position(|r| r == v).expect("read")])
                .collect();

            let value = body
                .apply(node.primitive.clone(), &operands)
                .expect("the node alone");

            body.set_outputs(&[value]).expect("its value");
            let by_value: Vec<bool> = reads.iter().map(|v| scalars.contains(v)).collect();
            let name = kernel(&body, &by_value);
            let label = graph.label(node);
            let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();

            map[node.output] = out
                .apply(Primitive::Fusion { name, label, body }, &reads)
                .expect("a fusion typed as the node");

            continue;
        }

        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");

        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
