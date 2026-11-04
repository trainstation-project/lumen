//! How TPU fuses attention's backward: its answers to the `FlashAttentionBackward` pass.

use crate::graph::{Graph, Node, Primitive, Var};
use crate::graph_compiler::common::attention::{self, AttentionBackwards, Backward};
use crate::graph_compiler::tpu::TpuModule;

/// Each attention backward runs as a dK/dV kernel and a dQ kernel.
impl AttentionBackwards for TpuModule<'_> {
    /// None: the TPU backward runs dK/dV and dQ as separate kernels.
    fn together(&self) -> impl Fn(&Backward) -> bool + 'static {
        |_| false
    }

    /// Names the backward fusion `flash_attention_backward`.
    fn backward_kernel(&mut self) -> String {
        "flash_attention_backward".into()
    }
}

/// Moves the dot computing a backward fusion's `delta` input inside the fusion body.
///
/// The fusion then reads the dot's operands, through any reshape/transpose chains, instead of `delta`.
pub(in crate::graph_compiler::common::attention) fn delta_inside(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let producer = |v: Var| nodes.iter().find(|n| n.output == v);

    // Walk back through reshapes and transposes to the value they were applied to.
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

            // Keep every body input except `delta`, which is now computed inside.
            for (j, &v) in body.inputs().iter().enumerate() {
                if j != k {
                    within[v] = inner.input(body.type_of(v).clone());
                    reads.push(map[node.inputs[j]]);
                    outer.push((node.inputs[j], within[v]));
                }
            }

            let mut sides = Vec::new();

            // Reuse an existing read when the dot's operand is already a fusion input.
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
