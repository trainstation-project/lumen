//! Plans which nodes fuse together into TPU `loop` and `reduction` kernels.
use crate::codegen::elementwise::tpu::grid::{layout, plan};
use crate::codegen::reduction::tpu::reducible;
use crate::codegen::tpu::fusible;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::common::loops::{body, members};

/// How a node may join the fusion group of its readers.
#[derive(Clone, Copy, PartialEq)]
enum Join {
    /// Joins its readers' group when the usual rules allow.
    Any,
    /// Starts its own group, which its producers may still join.
    Alone,
    /// Stays out of every fusion.
    Bare,
}

/// Fusion assignment of every node.
pub(super) struct Groups {
    /// Root node of the group each node belongs to, `None` for unfused nodes.
    pub(super) group: Vec<Option<usize>>,
    /// Per root, whether its group contains a reduction.
    pub(super) reduces: Vec<bool>,
}

/// Whether node `i` moves data by index (a layout op) and isn't plain elementwise.
fn indexes(graph: &Graph, i: usize) -> bool {
    let p = &graph.nodes()[i].primitive;
    layout(p) && !fusible(p)
}

/// Groups nodes back to front: each fusible node joins its readers' group when they all share one.
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

    // Walk consumers before producers so a producer can join the group of its readers.
    for i in (0..nodes.len()).rev() {
        let (reduction, indexed) = (reducing(i), indexes(graph, i));

        if join[i] == Join::Bare || !(fusible(&nodes[i].primitive) || reduction || indexed) {
            continue;
        }

        let v = nodes[i].output;
        let shape = &graph.type_of(v).shape;

        // A scalar fill only broadcast by its readers fuses regardless of shape.
        let constant = shape.is_empty()
            && matches!(nodes[i].primitive, Primitive::Full { .. })
            && readers[v]
                .iter()
                .all(|&r| matches!(nodes[r].primitive, Primitive::BroadcastInDim { .. }));

        let shaped = |r: usize| {
            reducing(r) || indexes(graph, r) || graph.type_of(nodes[r].output).shape == *shape
        };

        // Join only when every reader is in one group; a group holds at most one reduction and
        // never mixes a reduction with layout ops.
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

/// Groups nodes, splitting off layout ops until every loop group's indexing can be planned.
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

            // Groups whose indexing plans fine stay as they are.
            if layouts.is_empty() || plan(&body(graph, &members, root, &[]).0, 0).is_ok() {
                continue;
            }

            changed = true;
            // Isolate the offending layout ops: drop a lone root from fusion, otherwise give each
            // non-root layout op its own group.
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
