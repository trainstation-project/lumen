use super::TpuModule;
use crate::codegen::tpu::fusible;
use crate::graph::{FUSION_SEPARATOR, Graph, Primitive, Var, intern};
use crate::graph_compiler::pass::CompilerPass;

pub(super) struct LoopFusion;

impl CompilerPass<TpuModule<'_>> for LoopFusion {
    fn name(&self) -> &'static str {
        "loop_fusion"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = loop_fusion(&m.graph);
        Ok(())
    }
}

fn groups(graph: &Graph) -> Vec<Option<usize>> {
    let nodes = graph.nodes();
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        for &v in &node.inputs {
            if !readers[v].contains(&i) {
                readers[v].push(i);
            }
        }
    }
    let mut group: Vec<Option<usize>> = vec![None; nodes.len()];
    for i in (0..nodes.len()).rev() {
        if !fusible(&nodes[i].primitive) {
            continue;
        }
        let v = nodes[i].output;
        let shape = &graph.type_of(v).shape;
        let constant = shape.is_empty()
            && matches!(nodes[i].primitive, Primitive::Full { .. })
            && readers[v]
                .iter()
                .all(|&r| matches!(nodes[r].primitive, Primitive::BroadcastInDim { .. }));
        let joined = match readers[v].split_first() {
            Some((&first, rest)) if !graph.outputs().contains(&v) => group[first].filter(|&g| {
                rest.iter().all(|&r| group[r] == Some(g))
                    && (graph.type_of(nodes[g].output).shape == *shape || constant)
            }),
            _ => None,
        };
        group[i] = Some(joined.unwrap_or(i));
    }
    group
}

fn loop_fusion(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let group = groups(graph);
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        match group[i] {
            None => {
                let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(node.primitive.clone(), &inputs)
                    .expect("a node of the graph");
                out.set_label(map[node.output], node.label);
            }
            Some(g) if g == i => {
                let members: Vec<usize> = (0..=i).filter(|&j| group[j] == Some(i)).collect();
                let made: Vec<Var> = members.iter().map(|&j| nodes[j].output).collect();
                let mut reads: Vec<Var> = Vec::new();
                for &j in &members {
                    for &v in &nodes[j].inputs {
                        if !made.contains(&v) && !reads.contains(&v) {
                            reads.push(v);
                        }
                    }
                }
                let mut body = Graph::new();
                let mut within: Vec<Var> = vec![usize::MAX; graph.types.len()];
                for &v in &reads {
                    within[v] = body.input(graph.type_of(v).clone());
                }
                for &j in &members {
                    let ins: Vec<Var> = nodes[j].inputs.iter().map(|&v| within[v]).collect();
                    within[nodes[j].output] = body
                        .apply(nodes[j].primitive.clone(), &ins)
                        .expect("a member of the loop");
                }
                body.set_outputs(&[within[node.output]])
                    .expect("the loop's value");
                let labels: Vec<&str> = members.iter().map(|&j| graph.label(&nodes[j])).collect();
                let label = intern(labels.join(FUSION_SEPARATOR));
                let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(
                        Primitive::Fusion {
                            name: "loop".into(),
                            label,
                            body,
                        },
                        &reads,
                    )
                    .expect("a fusion typed as its root");
            }
            Some(_) => {}
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
