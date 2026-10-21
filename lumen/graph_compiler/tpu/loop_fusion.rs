use super::TpuModule;
use crate::codegen::tpu::fusible;
use crate::codegen::tpu::indexing::{layout, plan};
use crate::codegen::tpu::reduction::reducible;
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

#[derive(Clone, Copy, PartialEq)]
enum Join {
    Any,
    Alone,
    Bare,
}

struct Groups {
    group: Vec<Option<usize>>,
    reduces: Vec<bool>,
}

fn indexes(graph: &Graph, i: usize) -> bool {
    let p = &graph.nodes()[i].primitive;
    layout(p) && !fusible(p)
}

fn groups(graph: &Graph, join: &[Join]) -> Groups {
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
    let mut reduces = vec![false; nodes.len()];
    let mut layouts = vec![false; nodes.len()];
    let reducing = |r: usize| reducible(graph, &nodes[r]);
    for i in (0..nodes.len()).rev() {
        let (reduction, indexed) = (reducing(i), indexes(graph, i));
        if join[i] == Join::Bare || !(fusible(&nodes[i].primitive) || reduction || indexed) {
            continue;
        }
        let v = nodes[i].output;
        let shape = &graph.type_of(v).shape;
        let constant = shape.is_empty()
            && matches!(nodes[i].primitive, Primitive::Full { .. })
            && readers[v]
                .iter()
                .all(|&r| matches!(nodes[r].primitive, Primitive::BroadcastInDim { .. }));
        let shaped = |r: usize| {
            reducing(r) || indexes(graph, r) || graph.type_of(nodes[r].output).shape == *shape
        };
        let joined = match readers[v].split_first() {
            Some((&first, rest)) if join[i] == Join::Any && !graph.outputs().contains(&v) => {
                group[first].filter(|&g| {
                    rest.iter().all(|&r| group[r] == Some(g))
                        && (readers[v].iter().all(|&r| shaped(r)) || constant)
                        && !(reduction && (reduces[g] || layouts[g]))
                        && !(indexed && reduces[g])
                })
            }
            _ => None,
        };
        let g = joined.unwrap_or(i);
        group[i] = Some(g);
        reduces[g] |= reduction;
        layouts[g] |= indexed;
    }
    Groups { group, reduces }
}

fn members(group: &[Option<usize>], root: usize) -> Vec<usize> {
    (0..=root).filter(|&j| group[j] == Some(root)).collect()
}

fn body(graph: &Graph, members: &[usize], root: usize) -> (Graph, Vec<Var>) {
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

fn planned(graph: &Graph) -> Groups {
    let mut join = vec![Join::Any; graph.nodes().len()];
    loop {
        let groups = groups(graph, &join);
        let mut changed = false;
        for root in 0..join.len() {
            if groups.group[root] != Some(root) || groups.reduces[root] {
                continue;
            }
            let members = members(&groups.group, root);
            let layouts: Vec<usize> = members
                .iter()
                .copied()
                .filter(|&j| indexes(graph, j))
                .collect();
            if layouts.is_empty() || plan(&body(graph, &members, root).0, 0).is_ok() {
                continue;
            }
            changed = true;
            let joined: Vec<usize> = layouts.iter().copied().filter(|&j| j != root).collect();
            match joined.is_empty() {
                true => join[root] = Join::Bare,
                false => joined.iter().for_each(|&j| join[j] = Join::Alone),
            }
        }
        if !changed {
            return groups;
        }
    }
}

fn loop_fusion(graph: &Graph) -> Graph {
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
