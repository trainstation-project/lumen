use crate::codegen::tpu::fusible;
use crate::codegen::tpu::indexing::{layout, plan};
use crate::codegen::tpu::reduction::reducible;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::common::loops::{body, members};

#[derive(Clone, Copy, PartialEq)]
enum Join {
    Any,
    Alone,
    Bare,
}

pub(super) struct Groups {
    pub(super) group: Vec<Option<usize>>,
    pub(super) reduces: Vec<bool>,
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

pub(super) fn planned(graph: &Graph) -> Groups {
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
            if layouts.is_empty() || plan(&body(graph, &members, root, &[]).0, 0).is_ok() {
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
