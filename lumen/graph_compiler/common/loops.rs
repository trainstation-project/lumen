use crate::graph::{Graph, Primitive, Var};
use crate::graph_compiler::label;

pub(crate) fn members(group: &[Option<usize>], root: usize) -> Vec<usize> {
    (0..=root).filter(|&j| group[j] == Some(root)).collect()
}

pub(crate) fn body(
    graph: &Graph,
    members: &[usize],
    root: usize,
    extra: &[Var],
) -> (Graph, Vec<Var>) {
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
    let outputs: Vec<Var> = std::iter::once(nodes[root].output)
        .chain(extra.iter().copied())
        .map(|v| within[v])
        .collect();
    body.set_outputs(&outputs).expect("the loop's values");
    (body, reads)
}

pub(crate) fn fused(
    graph: &Graph,
    group: &[Option<usize>],
    kind: impl Fn(usize, &[usize]) -> (&'static str, usize),
    extra: impl Fn(usize) -> Vec<Var>,
) -> Graph {
    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
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
                let members = members(group, i);
                let extra = extra(i);
                let (body, reads) = body(graph, &members, i, &extra);
                let label = label(graph, &producer, &members);
                let (name, origin) = kind(i, &members);
                out.set_origin(&nodes[origin]);
                let reads: Vec<Var> = reads.iter().map(|&v| map[v]).collect();
                map[node.output] = out
                    .apply(
                        Primitive::Fusion {
                            name: name.into(),
                            label,
                            body,
                        },
                        &reads,
                    )
                    .expect("a fusion typed as its root");
                for (k, &v) in extra.iter().enumerate() {
                    let ty = graph.type_of(v).clone();
                    map[v] = out
                        .apply(
                            Primitive::FusionOutput { index: k + 1, ty },
                            &[map[node.output]],
                        )
                        .expect("the fusion's output");
                }
            }
            Some(_) => {}
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
