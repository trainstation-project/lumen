//! Common subexpression elimination (XLA: `HloCSE`): a node applying the
//! same primitive, parameters included, to the same values as an earlier
//! one is that node's value, computed once. Commutative primitives match
//! with their operands in either order. Graphs are pure, so this never
//! changes a result.

use std::collections::HashMap;

use crate::graph::{Graph, Primitive, Var};

/// Earlier nodes by primitive name and operands (sorted for commutative
/// primitives): their primitives, to compare parameters, and values.
type Seen = HashMap<(&'static str, Vec<Var>), Vec<(Primitive, Var)>>;

/// `graph` with each repeated computation replaced by its first.
pub(crate) fn cse(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut seen = Seen::new();
    for node in graph.nodes() {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        // A custom op's function is opaque: each call made, as written.
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
