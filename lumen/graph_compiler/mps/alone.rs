use super::{Kernel, MpsModule, request};
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct Alone;

impl<'a> CompilerPass<MpsModule<'a>> for Alone {
    fn name(&self) -> &'static str {
        "alone"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let graph = &m.graph;
        let by_value = |i: &usize| {
            let v = graph.inputs()[*i];
            let fused = |p: &Primitive| matches!(p, Primitive::Fusion { .. }) || alone(p);
            !graph.outputs().contains(&v)
                && graph
                    .nodes()
                    .iter()
                    .filter(|n| n.inputs.contains(&v))
                    .all(|n| fused(&n.primitive))
        };
        m.scalars.retain(by_value);
        let scalars: Vec<Var> = m.scalars.iter().map(|&i| graph.inputs()[i]).collect();
        let kernels = &mut m.kernels;
        m.graph = wrapped(graph, &scalars, alone, |by_value| {
            request(kernels, Kernel::Loop(by_value.to_vec()))
        });
        Ok(())
    }
}

fn alone(p: &Primitive) -> bool {
    matches!(
        p,
        Primitive::Concatenate {
            in_place: false,
            ..
        }
    ) || elementwise(p)
}

pub(super) fn wrapped(
    graph: &Graph,
    scalars: &[Var],
    wraps: impl Fn(&Primitive) -> bool,
    mut kernel: impl FnMut(&[bool]) -> String,
) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        if wraps(&node.primitive) {
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
            let name = kernel(&by_value);
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
