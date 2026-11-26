//! Pass joining a row gather with the `row` fusion that reads its rows: `gather(table, ids)` of a `[vocab, n]` table's
//! rows at int32 indices (an embedding lookup) feeding a row fusion (the first norm) becomes that fusion's first node,
//! so its kernel fetches the rows itself instead of a gather kernel writing them for the row to read back.
//!
//! - Matches a `Gather { axis: 0 }` of a 2-D table at int32 `ids` whose dims are all 1 but the last, its result read
//!   by a `row` fusion, directly or through one reshape adding or dropping size-1 dims (a lookup at flattened ids
//!   reshaped back to `[batch, seq, n]`) that only the row reads; no other reader of what the row reads is one the row
//!   depends on. Those readers then wait for the fusion.
//! - The joined body reads the table, the ids, then the row's other operands; it gathers (and reshapes), then runs the
//!   row's nodes. What the row read is one more output of the fusion (read back with `FusionOutput`) when something
//!   else reads it or the graph returns it (the residual stream).
//! - Each gather joins at most one row; the joined body must pass the device's `fits`.
//!
//! Devices: TPU runs it, with `normalization_diamonds`, right after `dot_rows` (`fits` is `codegen::row::tpu::fits`,
//! which takes a body opening with such a gather). Elsewhere a gather fuses into elementwise kernels already.

use crate::DType;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::CompilerPass;

/// A matched gather and row: the gather's node, the reshape between them if any, and the row's operand slot they feed.
type Joined = (usize, Option<usize>, usize);

/// Whether `node` gathers rows of a 2-D table at int32 indices whose dims are all 1 but the last.
fn row_gather(graph: &Graph, node: &Node) -> bool {
    let Primitive::Gather { axis: 0 } = node.primitive else {
        return false;
    };

    let (table, ids) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
    let rank = ids.shape.len();

    table.shape.len() == 2
        && ids.dtype == DType::I32
        && rank >= 1
        && ids.shape[..rank - 1].iter().all(|&d| d == 1)
}

/// Whether `node` is a reshape adding or dropping size-1 dims only.
fn unit_reshape(graph: &Graph, node: &Node) -> bool {
    let dims = |v: Var| -> Vec<usize> {
        graph
            .type_of(v)
            .shape
            .iter()
            .copied()
            .filter(|&d| d != 1)
            .collect()
    };

    matches!(node.primitive, Primitive::Reshape { .. }) && dims(node.inputs[0]) == dims(node.output)
}

/// The nodes `row` (node `r`) depends on, directly or not.
fn ancestors(graph: &Graph, producer: &[Option<usize>], r: usize) -> Vec<bool> {
    let nodes = graph.nodes();
    let mut seen = vec![false; nodes.len()];
    let mut stack = vec![r];

    while let Some(k) = stack.pop() {
        for &v in &nodes[k].inputs {
            if let Some(p) = producer[v]
                && !seen[p]
            {
                seen[p] = true;
                stack.push(p);
            }
        }
    }

    seen
}

/// The gather feeding `row` (node `r`), directly or through a unit reshape only the row reads, when the row depends on
/// no other reader of what it reads.
fn joined(graph: &Graph, producer: &[Option<usize>], (r, row): (usize, &Node)) -> Option<Joined> {
    let Primitive::Fusion { name, .. } = &row.primitive else {
        return None;
    };

    if name != "row" {
        return None;
    }

    let nodes = graph.nodes();

    let read_only_by = |v: Var, by: usize| {
        !graph.outputs().contains(&v)
            && nodes
                .iter()
                .enumerate()
                .all(|(k, n)| k == by || !n.inputs.contains(&v))
    };

    row.inputs.iter().enumerate().find_map(|(at, &v)| {
        let k = producer[v]?;

        let (g, reshape) = match unit_reshape(graph, &nodes[k]) {
            true => (producer[nodes[k].inputs[0]]?, Some(k)),
            false => (k, None),
        };

        let through = reshape.is_none_or(|k| read_only_by(nodes[k].inputs[0], k));

        let before = ancestors(graph, producer, r);

        let independent = nodes
            .iter()
            .enumerate()
            .all(|(k, n)| k == r || !before[k] || !n.inputs.contains(&v));

        let once = row.inputs.iter().filter(|&&u| u == v).count() == 1;

        (row_gather(graph, &nodes[g]) && through && independent && once).then_some((g, reshape, at))
    })
}

/// The values a matched pair reads: the table, the ids, then the row's other operands.
fn reads(graph: &Graph, (g, _, at): Joined, row: &Node) -> Vec<Var> {
    let gather = &graph.nodes()[g];

    [gather.inputs[0], gather.inputs[1]]
        .into_iter()
        .chain(
            row.inputs
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != at)
                .map(|(_, &v)| v),
        )
        .collect()
}

/// What the row reads: the reshape's value, or the gather's.
fn read(graph: &Graph, (g, reshape, _): Joined) -> Var {
    graph.nodes()[reshape.unwrap_or(g)].output
}

/// Whether what the row reads is read outside it, so the fusion writes it too.
fn kept(graph: &Graph, pair: Joined, row: &Node) -> bool {
    let v = read(graph, pair);

    graph.outputs().contains(&v)
        || graph
            .nodes()
            .iter()
            .any(|n| n.inputs.contains(&v) && !std::ptr::eq(n, row))
}

/// The joined body: the pair's reads in, the gather (and reshape), the row's nodes on it, the row's values out, then
/// what the row read when it is `kept`.
fn body(graph: &Graph, pair @ (g, reshape, at): Joined, row: &Node) -> Graph {
    let Primitive::Fusion { body: rows, .. } = &row.primitive else {
        unreachable!("a row fusion")
    };

    let nodes = graph.nodes();
    let mut out = Graph::new();

    let inputs: Vec<Var> = reads(graph, pair, row)
        .iter()
        .map(|&v| out.input(graph.type_of(v).clone()))
        .collect();

    let mut gathered = out
        .apply(nodes[g].primitive.clone(), &inputs[..2])
        .expect("the gather");

    if let Some(k) = reshape {
        gathered = out
            .apply(nodes[k].primitive.clone(), &[gathered])
            .expect("the reshape");
    }

    let mut others = inputs[2..].iter().copied();
    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];

    // Slot `at` of the row now reads the gathered rows; the rest map to the new inputs in order.
    for (k, &v) in rows.inputs().iter().enumerate() {
        map[v] = match k == at {
            true => gathered,
            false => others.next().expect("a row's operand"),
        };
    }

    for node in rows.nodes() {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the row");
    }

    let outputs: Vec<Var> = rows
        .outputs()
        .iter()
        .map(|&v| map[v])
        .chain(kept(graph, pair, row).then_some(gathered))
        .collect();

    out.set_outputs(&outputs).expect("the row's values");
    out
}

/// `graph` with each row gather joined to the row fusion reading its rows, where the device `fits` the joined body.
pub(crate) fn gather_rows(graph: &Graph, fits: fn(&Graph) -> bool) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let mut gather_of: Vec<Option<Joined>> = vec![None; nodes.len()];
    let mut taken = vec![false; nodes.len()];

    for (r, row) in nodes.iter().enumerate() {
        if let Some(pair @ (g, reshape, _)) = joined(graph, &producer, (r, row))
            && !taken[g]
            && fits(&body(graph, pair, row))
        {
            taken[g] = true;
            reshape.into_iter().for_each(|k| taken[k] = true);
            gather_of[r] = Some(pair);
        }
    }

    if gather_of.iter().all(Option::is_none) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // Emit nodes in order, deferring any whose reads aren't mapped yet: a reader of a joined value emitted before its
    // row now reads the fusion's output, so it waits for it.
    let mut deferred: Vec<usize> = Vec::new();

    for (i, &inside) in taken.iter().enumerate() {
        // Taken gathers (and reshapes) are emitted inside the fusion of their row.
        if inside {
            continue;
        }

        deferred.push(i);

        while let Some(k) = deferred.iter().position(|&j| {
            let reads = match gather_of[j] {
                Some(pair) => reads(graph, pair, &nodes[j]),
                None => nodes[j].inputs.clone(),
            };

            reads.iter().all(|&v| map[v] != usize::MAX)
        }) {
            let j = deferred.remove(k);
            emit(&mut out, &mut map, graph, j, gather_of[j]);
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

/// Emits node `i` of `graph` into `out`: the joined fusion of its row and gather when `joined`, else the node itself.
fn emit(out: &mut Graph, map: &mut [Var], graph: &Graph, i: usize, joined: Option<Joined>) {
    let nodes = graph.nodes();
    let node = &nodes[i];
    out.set_origin(node);

    let Some(pair @ (g, reshape, _)) = joined else {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the graph");

        out.set_label(map[node.output], node.label);
        return;
    };

    let read_values: Vec<Var> = reads(graph, pair, node).iter().map(|&v| map[v]).collect();

    let label = std::iter::once(&nodes[g])
        .chain(reshape.map(|k| &nodes[k]))
        .chain([node])
        .map(|n| graph.label(n))
        .collect::<Vec<_>>()
        .join(FUSION_SEPARATOR);

    let fused = Primitive::Fusion {
        name: "row".into(),
        label: intern(label),
        body: body(graph, pair, node),
    };

    let value = out
        .apply(fused, &read_values)
        .expect("a fusion typed as its row");

    map[node.output] = value;

    if kept(graph, pair, node) {
        let Primitive::Fusion { body: rows, .. } = &node.primitive else {
            unreachable!("a row fusion")
        };

        let v = read(graph, pair);
        let ty = graph.type_of(v).clone();
        let index = rows.outputs().len();

        map[v] = out
            .apply(Primitive::FusionOutput { index, ty }, &[value])
            .expect("the gathered value");
    }
}

/// Joins row gathers with the row fusions reading them, keeping the joined bodies the device `fits`.
pub(crate) struct GatherRows {
    /// Whether the device's row kernel runs a joined body.
    pub(crate) fits: fn(&Graph) -> bool,
}

impl<D> CompilerPass<DeviceModule<'_, D>> for GatherRows {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "gather_rows"
    }

    /// Joins the module's row gathers with the row fusions reading them.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        m.graph = gather_rows(&m.graph, self.fits);
        Ok(())
    }
}
