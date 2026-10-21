use super::TpuModule;
use crate::codegen::tpu::row::{fits, fusible};
use crate::graph::{FUSION_SEPARATOR, Graph, Primitive, Var, intern};
use crate::graph_compiler::diamonds::{Row, diamonds};
use crate::graph_compiler::pass::CompilerPass;

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

struct Fused {
    members: Vec<usize>,
    reads: Vec<Var>,
    outputs: Vec<Var>,
    body: Graph,
}

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

fn fused(graph: &Graph, rows: &[Row]) -> Graph {
    let nodes = graph.nodes();
    let mut root: Vec<Option<Fused>> = (0..nodes.len()).map(|_| None).collect();
    let mut member = vec![false; nodes.len()];
    for row in rows {
        let f = fusion(graph, row);
        if !fits(&f.body) {
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
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        if let Some(f) = root[i].take() {
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
            continue;
        }
        if member[i] {
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
