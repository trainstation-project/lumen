//! Fuses normalization diamonds (row reductions broadcast back over their rows) into `row` kernels.
use super::TpuModule;
use crate::codegen::tpu::row::{fits, fusible};
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::common::diamonds::{Row, diamonds};
use crate::graph_compiler::pass::CompilerPass;

/// Replaces normalization diamonds the TPU row kernel supports with `row` fusions.
pub(super) struct Diamonds;

impl CompilerPass<TpuModule<'_>> for Diamonds {
    fn name(&self) -> &'static str {
        "normalization_diamonds"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = fused(&m.graph, &diamonds(&m.graph, fusible));
        Ok(())
    }
}

/// A candidate `row` fusion extracted from a diamond.
struct Fused {
    /// Sorted node indices folded into the fusion.
    members: Vec<usize>,
    /// Values read from outside the fusion, in body input order.
    reads: Vec<Var>,
    /// Values the fusion produces: the root's first, then the row's extra outputs.
    outputs: Vec<Var>,
    body: Graph,
}

/// Collects the members, external reads, outputs and body graph of `row`.
fn fusion(graph: &Graph, row: &Row) -> Fused {
    let nodes = graph.nodes();
    let mut members: Vec<usize> = row
        .inner
        .iter()
        .chain(&row.reductions)
        .chain(&row.partials)
        .chain([&row.root])
        .copied()
        .collect();
    members.sort_unstable();
    members.dedup();
    let made: Vec<Var> = members.iter().map(|&j| nodes[j].output).collect();
    let mut reads: Vec<Var> = Vec::new();
    for &j in &members {
        for &v in &nodes[j].inputs {
            if !made.contains(&v) && !reads.contains(&v) {
                reads.push(v);
            }
        }
    }
    let outputs: Vec<Var> = std::iter::once(nodes[row.root].output)
        .chain(row.outputs.iter().map(|&k| nodes[k].output))
        .collect();
    let mut body = Graph::new();
    let mut within: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in &reads {
        within[v] = body.input(graph.type_of(v).clone());
    }
    for &j in &members {
        let ins: Vec<Var> = nodes[j].inputs.iter().map(|&v| within[v]).collect();
        within[nodes[j].output] = body
            .apply(nodes[j].primitive.clone(), &ins)
            .expect("a member of the row");
    }
    let values: Vec<Var> = outputs.iter().map(|&v| within[v]).collect();
    body.set_outputs(&values).expect("the row's values");
    Fused {
        members,
        reads,
        outputs,
        body,
    }
}

/// Whether no read of `f` depends on one of its members, so the fusion can be emitted as one node.
fn convex(graph: &Graph, f: &Fused) -> bool {
    let mut after = vec![false; graph.types.len()];
    for (i, node) in graph.nodes().iter().enumerate() {
        let member = f.members.binary_search(&i).is_ok();
        after[node.output] = member || node.inputs.iter().any(|&v| after[v]);
    }
    !f.reads.iter().any(|&v| after[v])
}

/// Whether every member value not exposed as an output is read only inside the fusion.
fn sealed(graph: &Graph, f: &Fused) -> bool {
    let inside = |v: Var| {
        let readers = graph
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| n.inputs.contains(&v));
        !graph.outputs().contains(&v)
            && readers
                .map(|(k, _)| k)
                .all(|k| f.members.binary_search(&k).is_ok())
    };
    let nodes = graph.nodes();
    f.members
        .iter()
        .map(|&j| nodes[j].output)
        .all(|v| f.outputs.contains(&v) || inside(v))
}

/// Rewrites `graph` with each valid row fusion replacing its members.
pub(super) fn fused(graph: &Graph, rows: &[Row]) -> Graph {
    let nodes = graph.nodes();
    let mut root: Vec<Option<Fused>> = (0..nodes.len()).map(|_| None).collect();
    let mut member = vec![false; nodes.len()];
    for row in rows {
        let f = fusion(graph, row);
        if !fits(&f.body) || !convex(graph, &f) || !sealed(graph, &f) {
            continue;
        }
        f.members.iter().for_each(|&j| member[j] = true);
        root[row.root] = Some(f);
    }
    if root.iter().all(Option::is_none) {
        return graph.clone();
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    // Emit nodes in order but defer any whose reads aren't mapped yet: a fusion emitted at its root
    // may read values made later in the graph, and its readers then wait for it.
    let mut deferred: Vec<usize> = Vec::new();
    for i in 0..nodes.len() {
        if member[i] && root[i].is_none() {
            continue;
        }
        deferred.push(i);
        while let Some(k) = deferred.iter().position(|&j| {
            let reads = root[j].as_ref().map_or(&nodes[j].inputs, |f| &f.reads);
            reads.iter().all(|&v| map[v] != usize::MAX)
        }) {
            let j = deferred.remove(k);
            match root[j].take() {
                Some(f) => emit(&mut out, &mut map, graph, &nodes[j], f),
                None => copy(&mut out, &mut map, &nodes[j]),
            }
        }
    }
    assert!(
        deferred.is_empty(),
        "every node reads values computed before it"
    );
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Copies `node` unchanged into `out`.
fn copy(out: &mut Graph, map: &mut [Var], node: &Node) {
    out.set_origin(node);
    let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
    map[node.output] = out
        .apply(node.primitive.clone(), &inputs)
        .expect("a node of the graph");
    out.set_label(map[node.output], node.label);
}

/// Emits `f` as a `row` fusion, with extra outputs exposed through `FusionOutput` nodes.
fn emit(out: &mut Graph, map: &mut [Var], graph: &Graph, node: &Node, f: Fused) {
    out.set_origin(node);
    let nodes = graph.nodes();
    let labels: Vec<&str> = f.members.iter().map(|&j| graph.label(&nodes[j])).collect();
    let reads: Vec<Var> = f.reads.iter().map(|&v| map[v]).collect();
    let fused = out
        .apply(
            Primitive::Fusion {
                name: "row".into(),
                label: intern(labels.join(FUSION_SEPARATOR)),
                body: f.body,
            },
            &reads,
        )
        .expect("a fusion typed as its root");
    map[f.outputs[0]] = fused;
    for (index, &v) in f.outputs.iter().enumerate().skip(1) {
        let ty = graph.type_of(v).clone();
        map[v] = out
            .apply(Primitive::FusionOutput { index, ty }, &[fused])
            .expect("the row's output");
    }
}
