use super::{Kernel, MpsModule, request};
use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct ConcatenatesAlone;

impl<'a> CompilerPass<MpsModule<'a>> for ConcatenatesAlone {
    fn name(&self) -> &'static str {
        "concatenates_alone"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        m.scalars.clear();
        let kernels = &mut m.kernels;
        m.graph = concatenates_alone(&m.graph, |_| request(kernels, Kernel::Loop(Vec::new())));
        Ok(())
    }
}

pub(super) fn concatenates_alone(graph: &Graph, mut kernel: impl FnMut(&Graph) -> String) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for node in graph.nodes() {
        out.set_origin(node);
        let primitive = match node.primitive {
            Primitive::Concatenate {
                in_place: false, ..
            } => {
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
                    .expect("the concatenate");
                body.set_outputs(&[value]).expect("its value");
                let name = kernel(&body);
                let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(
                        Primitive::Fusion {
                            name,
                            label: "concatenate",
                            body,
                        },
                        &reads,
                    )
                    .expect("a fusion typed as the concatenate");
                continue;
            }
            ref p => p.clone(),
        };
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out.apply(primitive, &inputs).expect("a node of the graph");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
