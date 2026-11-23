//! Fusion by planned groups (TPU's): plans which nodes fuse together into `elementwise` and `reduction` kernels,
//! splitting off layout ops until every group's indexing can be planned, and turns the groups into fusions, a
//! reducing member leading. `codegen::fusion::tpu` runs it as TPU's fusion.
use crate::codegen::devices::tpu::fusible;
use crate::codegen::elementwise::tpu::grid::{layout, plan};
use crate::codegen::reduction::tpu::reducible;
use crate::graph::{Graph, Primitive};
use crate::graph_compiler::passes::elementwise::{body, fused, members};

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
struct Groups {
    /// Root node of the group each node belongs to, `None` for unfused nodes.
    group: Vec<Option<usize>>,
    /// Per root, whether its group contains a reduction.
    reduces: Vec<bool>,
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

/// Groups nodes but the `held` ones (a small fusion's), splitting off layout ops until every elementwise group's
/// indexing can be planned.
fn planned(graph: &Graph, held: &[bool]) -> Groups {
    let mut join: Vec<Join> = held
        .iter()
        .map(|&h| if h { Join::Bare } else { Join::Any })
        .collect();

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

/// Each node's small fusion (`codegen::elementwise::tpu::small`), by its root (its last node): the connected groups
/// of two or more nodes the small kernel takes, each reading only graph inputs and other such nodes (so the group is
/// computed from inputs alone), whose inputs and outputs it reads and writes whole, all joined into one fusion.
fn small_groups(graph: &Graph) -> Vec<Option<usize>> {
    use crate::codegen::elementwise::tpu::small::{buffer, takes};

    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }

    let mut ok = vec![false; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        ok[i] = takes(graph, node)
            && node
                .inputs
                .iter()
                .all(|&v| producer[v].is_none_or(|k| ok[k]));
    }

    // Connected groups of taken nodes, by union-find.
    let mut parent: Vec<usize> = (0..nodes.len()).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    for (i, node) in nodes.iter().enumerate().filter(|&(i, _)| ok[i]) {
        for &v in &node.inputs {
            if let Some(k) = producer[v].filter(|&k| ok[k]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, k));
                parent[a] = b;
            }
        }
    }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];

    for i in (0..nodes.len()).filter(|&i| ok[i]) {
        let r = find(&mut parent, i);
        members[r].push(i);
    }

    let mut group = vec![None; nodes.len()];

    for m in members.iter().filter(|m| m.len() >= 2) {
        let inside = |k: usize| m.contains(&k);

        let reads = m
            .iter()
            .flat_map(|&k| &nodes[k].inputs)
            .filter(|&&v| producer[v].is_none_or(|k| !inside(k)));

        let written = m
            .iter()
            .map(|&k| nodes[k].output)
            .filter(|&v| graph.outputs().contains(&v) || readers[v].iter().any(|&r| !inside(r)));

        let whole = reads.clone().all(|&v| buffer(graph.type_of(v), false))
            && written.clone().all(|v| buffer(graph.type_of(v), true))
            && written.count() > 0;

        if whole {
            m.iter().for_each(|&k| group[k] = Some(0));
        }
    }

    // The groups read graph inputs alone, so they run as one kernel, rooted at their last node.
    let root = (0..nodes.len()).rev().find(|&k| group[k].is_some());
    group
        .iter_mut()
        .flatten()
        .for_each(|r| *r = root.expect("a member"));
    group
}

/// `graph` with the nodes `group` assigns a group moved to its start, in order (they read graph inputs alone).
fn first(graph: &Graph, group: &[Option<usize>]) -> Graph {
    if group.iter().all(Option::is_none) {
        return graph.clone();
    }

    let nodes = graph.nodes();
    let mut out = Graph::new();
    let mut map: Vec<usize> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    let order = (0..nodes.len())
        .filter(|&k| group[k].is_some())
        .chain((0..nodes.len()).filter(|&k| group[k].is_none()));

    for k in order {
        let node = &nodes[k];
        out.set_origin(node);
        let inputs: Vec<usize> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");
        out.set_label(map[node.output], node.label);
    }

    let outputs: Vec<usize> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Rewrites `graph` with each planned group replaced by one fusion, named `reduction` when it reduces, and each small
/// group (`small_groups`) by a `small` fusion.
pub(crate) fn elementwise_fusion(graph: &Graph) -> Graph {
    // Small groups read graph inputs alone: they move to the start, so each fusion (at its last node) precedes every
    // reader of its values.
    let moved = first(graph, &small_groups(graph));
    let graph = &moved;
    let nodes = graph.nodes();
    let small = small_groups(graph);
    let held: Vec<bool> = small.iter().map(Option::is_some).collect();
    let Groups { mut group, reduces } = planned(graph, &held);

    for (i, &s) in small.iter().enumerate() {
        if s.is_some() {
            group[i] = s;
        }
    }

    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        node.inputs.iter().for_each(|&v| readers[v].push(i));
    }

    // A slice alone in its group stays out: `place_views` may make it a view, else `alone` wraps it.
    for (i, node) in nodes.iter().enumerate() {
        if group[i] == Some(i)
            && matches!(node.primitive, Primitive::Slice { .. })
            && members(&group, i).len() == 1
        {
            group[i] = None;
        }
    }

    fused(
        graph,
        &group,
        |root, members| {
            // A reducing member leads the fusion so the kernel is built around it.
            let main = members.iter().find(|&&j| reducible(graph, &nodes[j]));

            let name = match (small[root].is_some(), reduces[root]) {
                (true, _) => crate::codegen::elementwise::tpu::small::SMALL,
                (_, true) => "reduction",
                _ => "elementwise",
            };

            (name, *main.unwrap_or(&root))
        },
        // A small group's other values read outside it are its further outputs.
        |root| match small[root] {
            Some(_) => members(&group, root)
                .into_iter()
                .filter(|&k| k != root)
                .map(|k| nodes[k].output)
                .filter(|&v| {
                    graph.outputs().contains(&v)
                        || readers[v].iter().any(|&r| group[r] != Some(root))
                })
                .collect(),
            None => Vec::new(),
        },
    )
}
