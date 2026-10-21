use std::collections::HashMap;

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

type Seen = HashMap<(&'static str, Vec<Var>), Vec<(Primitive, Var)>>;

pub(crate) fn cse(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut seen = Seen::new();
    for node in graph.nodes() {
        out.set_origin(node);
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        if let Primitive::CustomCall { .. } = node.primitive {
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("the node's own operands");
            continue;
        }
        let mut key = inputs.clone();
        if commutative(&node.primitive) {
            key.sort_unstable();
        }
        let earlier = seen.entry((node.primitive.name(), key)).or_default();
        if let Some(&(_, v)) = earlier.iter().find(|(p, _)| *p == node.primitive) {
            map[node.output] = v;
            continue;
        }
        let v = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the node's own operands");
        out.set_label(v, node.label);
        earlier.push((node.primitive.clone(), v));
        map[node.output] = v;
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

fn commutative(p: &Primitive) -> bool {
    use Primitive::*;
    matches!(p, Add | Mul | Max | Eq)
}

pub(crate) struct Cse;

impl<M: Module> CompilerPass<M> for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = cse(m.graph());
        Ok(())
    }
}
