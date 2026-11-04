//! Pass dropping a value's round trip through another float dtype (`cast(cast(x, b), a)`, `x` of dtype
//! `a`) where the forward reads it (`preserve_precision`): forward readers read `x` itself, unrounded.
//! A gradient's reader (a node autograd's backward pass traced, reading an activation saved for it) keeps
//! the narrower dtype: the round trip when `b` is narrower (the activation saved in `b`), `x` when `a`
//! is. Off by default: the forward then computes more precisely than the traced program.

use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};

/// For each value, the value it round-trips (`x` for `cast(cast(x, b), a)` with `x` of dtype `a`, both
/// casts between float dtypes), and whether the trip goes through a narrower dtype.
fn round_trips(graph: &Graph) -> Vec<Option<(Var, bool)>> {
    let nodes = graph.nodes();
    let mut producer = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let mut trips = vec![None; graph.types.len()];

    for node in nodes {
        let Primitive::Cast { new_dtype: a } = node.primitive else {
            continue;
        };

        let Some(inner) = producer[node.inputs[0]].map(|p| &nodes[p]) else {
            continue;
        };

        let Primitive::Cast { new_dtype: b } = inner.primitive else {
            continue;
        };

        let x = inner.inputs[0];

        if graph.type_of(x).dtype == a && a.is_float() && b.is_float() && a != b {
            trips[node.output] = Some((x, b.size_of() < a.size_of()));
        }
    }

    trips
}

/// `graph` with each round trip through another float dtype dropped where the forward reads it (and
/// where a gradient reads it, unless the trip is through a narrower dtype, which the gradient keeps).
pub(crate) fn preserve_precision(graph: &Graph) -> Graph {
    let trips = round_trips(graph);

    if trips.iter().all(Option::is_none) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // What a reader reads for `v`: the round-tripped value, unless it computes a gradient and the trip
    // is through a narrower dtype (the activation saved for it).
    let read = |v: Var, backward: bool| match trips[v] {
        Some((_, narrower)) if backward && narrower => v,
        Some((x, _)) => x,
        None => v,
    };

    for node in graph.nodes() {
        out.set_origin(node);
        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .map(|&v| map[read(v, node.backward)])
            .collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("well typed");
    }

    // The program's outputs are the forward's.
    let outputs: Vec<Var> = graph
        .outputs()
        .iter()
        .map(|&v| map[read(v, false)])
        .collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out.prune().0
}

/// Compiler pass running [`preserve_precision`].
pub(crate) struct PreservePrecision;

impl<M: Module> CompilerPass<M> for PreservePrecision {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "preserve_precision"
    }

    /// Replaces the module's graph with its round trips dropped.
    fn apply(&mut self, m: &mut M) -> Result<(), String> {
        *m.graph_mut() = preserve_precision(m.graph());
        Ok(())
    }
}
