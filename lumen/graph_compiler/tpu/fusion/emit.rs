use super::groups::{Groups, planned};
use crate::graph::{FUSION_SEPARATOR, Graph, Primitive, Var, intern};

pub(super) fn members(group: &[Option<usize>], root: usize) -> Vec<usize> {
    (0..=root).filter(|&j| group[j] == Some(root)).collect()
}

pub(super) fn body(graph: &Graph, members: &[usize], root: usize) -> (Graph, Vec<Var>) {
    let nodes = graph.nodes();
    let made: Vec<Var> = members.iter().map(|&j| nodes[j].output).collect();
    let mut reads: Vec<Var> = Vec::new();
    for &j in members {
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
    for &j in members {
        let ins: Vec<Var> = nodes[j].inputs.iter().map(|&v| within[v]).collect();
        within[nodes[j].output] = body
            .apply(nodes[j].primitive.clone(), &ins)
            .expect("a member of the loop");
    }
    body.set_outputs(&[within[nodes[root].output]])
        .expect("the loop's value");
    (body, reads)
}

pub(super) fn loop_fusion(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let Groups { group, reduces } = planned(graph);
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
                let members = members(&group, i);
                let (body, reads) = body(graph, &members, i);
                let labels: Vec<&str> = members.iter().map(|&j| graph.label(&nodes[j])).collect();
                let label = intern(labels.join(FUSION_SEPARATOR));
                let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(
                        Primitive::Fusion {
                            name: match reduces[i] {
                                true => "reduction".into(),
                                false => "loop".into(),
                            },
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
