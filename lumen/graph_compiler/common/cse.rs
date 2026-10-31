//! Common subexpression elimination.

use std::collections::HashMap;

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// Nodes already emitted, keyed by primitive name and (remapped) inputs; the full primitive
/// disambiguates nodes sharing a name but differing in parameters.
type Seen = HashMap<(&'static str, Vec<Var>), Vec<(Primitive, Var)>>;

/// Rebuilds `graph` with duplicate nodes (same primitive and inputs, commutative inputs in any order)
/// merged into one. Custom calls are never merged since they have side effects, nor sends and receives,
/// each of which is its own message.
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

        if matches!(node.primitive, Primitive::CustomCall { .. })
            || node.primitive.is_point_to_point()
        {
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

/// Primitives whose two operands can be swapped.
fn commutative(p: &Primitive) -> bool {
    use Primitive::*;
    matches!(p, Add | Mul | Max | Eq)
}

/// Compiler pass running [`cse`].
pub(crate) struct Cse;

impl<M: Module> CompilerPass<M> for Cse {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "cse"
    }

    /// Replaces the module's graph with its deduplicated version.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = cse(m.graph());
        Ok(())
    }
}
