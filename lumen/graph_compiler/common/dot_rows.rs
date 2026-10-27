//! Fuses a matmul into the normalization `row` fusion that consumes its whole output rows, as one
//! `dot_row` fusion: the row computed from the product without it reaching memory. With `through`, one
//! elementwise op may sit between them (a residual added to the product), its value written too where
//! it is read again.

use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::common::diamonds::tile_reductions;
use crate::graph_compiler::elementwise;

/// Whether a row fusion's body reads its input `at` as whole rows, so a matmul can feed it.
pub(crate) type Joins = fn(&Graph, usize) -> bool;

/// Whether `dot` is an unbatched matmul of a rank-2+ lhs contracted on its last axis with a 2-D rhs.
fn plain(graph: &Graph, dot: &Node) -> bool {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        ..
    } = &dot.primitive
    else {
        return false;
    };
    let (lhs, rhs) = (graph.type_of(dot.inputs[0]), graph.type_of(dot.inputs[1]));
    lhs_batch.is_empty()
        && lhs.shape.len() >= 2
        && *lhs_contracting == [lhs.shape.len() - 1]
        && rhs.shape.len() == 2
        && rhs_contracting.len() == 1
}

/// A matched matmul and row: the dot's node, the elementwise node between them if any, and the row's
/// operand slot the result feeds.
type Joined = (usize, Option<usize>, usize);

/// Finds the plain dot feeding `row` (node `r`) as an operand the backend `joins`, directly or, with
/// `through`, through one elementwise node whose other readers all come after the row.
fn joined(
    graph: &Graph,
    producer: &[Option<usize>],
    (r, row): (usize, &Node),
    joins: Joins,
    through: bool,
) -> Option<Joined> {
    let Primitive::Fusion { name, body, .. } = &row.primitive else {
        return None;
    };
    if name != "row" {
        return None;
    }
    let nodes = graph.nodes();
    let readers = |v: Var| {
        nodes
            .iter()
            .enumerate()
            .filter(move |(_, n)| n.inputs.contains(&v))
    };
    // The dot's result must be read whole by the row kernel, by nothing else, and not escape the graph.
    let only =
        |v: Var, by: usize| readers(v).all(|(k, _)| k == by) && !graph.outputs().contains(&v);
    row.inputs.iter().enumerate().find_map(|(at, &v)| {
        if !joins(body, at) {
            return None;
        }
        let k = producer[v]?;
        if plain(graph, &nodes[k]) && only(v, r) {
            return Some((k, None, at));
        }
        let e = &nodes[k];
        let fed = |&u: &Var| producer[u].is_some_and(|d| plain(graph, &nodes[d]) && only(u, k));
        let dot = e.inputs.iter().find(|u| fed(u))?;
        let later = readers(v).all(|(k, _)| k >= r);
        let same = graph.type_of(*dot).shape == graph.type_of(v).shape;
        (through && elementwise(&e.primitive) && later && same)
            .then(|| (producer[*dot].expect("a dot"), Some(k), at))
    })
}

/// The values a matched pair reads: the dot's operands, the linking op's others, the row's others.
fn reads(graph: &Graph, (d, link, at): Joined, row: &Node) -> Vec<Var> {
    let nodes = graph.nodes();
    let dot = &nodes[d];
    let linked = link
        .into_iter()
        .flat_map(|e| nodes[e].inputs.iter().copied().filter(|&u| u != dot.output));
    [dot.inputs[0], dot.inputs[1]]
        .into_iter()
        .chain(linked)
        .chain(
            row.inputs
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != at)
                .map(|(_, &v)| v),
        )
        .collect()
}

/// Whether the linking op's value is read outside the pair, so the fusion writes it too.
fn kept(graph: &Graph, link: Option<usize>, row: &Node) -> Option<Var> {
    let v = graph.nodes()[link?].output;
    let outside = graph
        .nodes()
        .iter()
        .any(|n| n.inputs.contains(&v) && !std::ptr::eq(n, row));
    (outside || graph.outputs().contains(&v)).then_some(v)
}

/// Applies a matched pair's dot (and linking op) in `out` to its read `inputs`; returns the value the
/// row reads and the index of the first input left for the row.
fn product(out: &mut Graph, graph: &Graph, (d, link, _): Joined, inputs: &[Var]) -> (Var, usize) {
    let nodes = graph.nodes();
    let dot = &nodes[d];
    let mut product = out
        .apply(dot.primitive.clone(), &inputs[..2])
        .expect("the dot");
    let mut next = 2;
    if let Some(e) = link {
        let operands: Vec<Var> = nodes[e]
            .inputs
            .iter()
            .map(|&u| match u == dot.output {
                true => product,
                false => {
                    next += 1;
                    inputs[next - 1]
                }
            })
            .collect();
        product = out
            .apply(nodes[e].primitive.clone(), &operands)
            .expect("the linking op");
    }
    (product, next)
}

/// Builds the `dot_row` body: the pair's reads in, the row's values out, then the linking op's value
/// when it is `kept`.
fn body(graph: &Graph, (d, link, at): Joined, row: &Node) -> Graph {
    let Primitive::Fusion { body: rows, .. } = &row.primitive else {
        unreachable!("a row fusion")
    };
    let mut out = Graph::new();
    let read = reads(graph, (d, link, at), row);
    let inputs: Vec<Var> = read
        .iter()
        .map(|&v| out.input(graph.type_of(v).clone()))
        .collect();
    let (product, next) = product(&mut out, graph, (d, link, at), &inputs);
    let linked = product;
    let others: Vec<Var> = inputs[next..].to_vec();
    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];
    let mut others = others.into_iter();
    // Slot `at` of the row now reads the in-body product; the rest map to the new inputs in order.
    for (k, &v) in rows.inputs().iter().enumerate() {
        map[v] = match k == at {
            true => product,
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
        .chain(kept(graph, link, row).map(|_| linked))
        .collect();
    out.set_outputs(&outputs).expect("the row's values");
    out
}

/// Replaces each matched row fusion and its feeding dot (and, with `through`, the elementwise op between
/// them) with one `dot_row` fusion whose body the backend `fits`.
pub(crate) fn dot_rows(
    graph: &Graph,
    joins: Joins,
    fits: fn(&Graph) -> bool,
    through: bool,
) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut dot_of: Vec<Option<Joined>> = vec![None; nodes.len()];
    let mut taken = vec![false; nodes.len()];
    for (r, row) in nodes.iter().enumerate() {
        if let Some((d, link, at)) = joined(graph, &producer, (r, row), joins, through)
            && !taken[d]
            && link.is_none_or(|e| !taken[e])
            && fits(&body(graph, (d, link, at), row))
        {
            taken[d] = true;
            link.into_iter().for_each(|e| taken[e] = true);
            dot_of[r] = Some((d, link, at));
        }
    }
    if dot_of.iter().all(Option::is_none) {
        return graph.clone();
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        // Taken dots are emitted inside the `dot_row` fusion of their row.
        if taken[i] {
            continue;
        }
        out.set_origin(node);
        if let Some(pair @ (d, link, _)) = dot_of[i] {
            let read: Vec<Var> = reads(graph, pair, node).iter().map(|&v| map[v]).collect();
            let label = std::iter::once(&nodes[d])
                .chain(link.map(|e| &nodes[e]))
                .chain([node])
                .map(|n| graph.label(n))
                .collect::<Vec<_>>()
                .join(FUSION_SEPARATOR);
            let fused = Primitive::Fusion {
                name: "dot_row".into(),
                label: intern(label),
                body: body(graph, pair, node),
            };
            let value = out.apply(fused, &read).expect("a fusion typed as its row");
            map[node.output] = value;
            if let Some(v) = kept(graph, link, node) {
                let Primitive::Fusion { body: rows, .. } = &node.primitive else {
                    unreachable!("a row fusion")
                };
                let ty = graph.type_of(v).clone();
                let index = rows.outputs().len();
                map[v] = out
                    .apply(Primitive::FusionOutput { index, ty }, &[value])
                    .expect("the linking op's value");
            }
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

/// The row's last-axis reductions in its body, in order.
fn row_reductions(body: &Graph) -> Vec<&Node> {
    body.nodes()
        .iter()
        .filter(|n| match &n.primitive {
            Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
                let rank = body.type_of(n.inputs[0]).shape.len();
                *axes == [rank.saturating_sub(1)]
            }
            _ => false,
        })
        .collect()
}

/// The reduction `node` performs, over `axes` instead.
fn over(node: &Node, axes: Vec<usize>) -> Primitive {
    match &node.primitive {
        Primitive::ReduceSum { accum_dtype, .. } => Primitive::ReduceSum {
            axes,
            accum_dtype: *accum_dtype,
        },
        _ => Primitive::ReduceMax { axes },
    }
}

/// A wide pair's split: the matched pair, the partial tile width, and the row body's reductions taken
/// partially (their outputs in the row body).
type Split = (Joined, usize, Vec<Var>);

/// Builds the `dot_partials` body: the pair's reads in; the value the row reads, then, for each split
/// reduction, its partials over `tile`-column blocks of each row (`[rows, n / tile]`), out.
fn partials_body(graph: &Graph, (pair, tile, split): &Split, row: &Node) -> Graph {
    let Primitive::Fusion { body: rows, .. } = &row.primitive else {
        unreachable!("a row fusion")
    };
    let at = pair.2;
    let mut out = Graph::new();
    let inputs: Vec<Var> = reads(graph, *pair, row)
        .iter()
        .map(|&v| out.input(graph.type_of(v).clone()))
        .collect();
    let (product, next) = product(&mut out, graph, *pair, &inputs);
    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];
    let mut others = inputs[next..].iter();
    for (k, &v) in rows.inputs().iter().enumerate() {
        map[v] = match k == at {
            true => product,
            false => *others.next().expect("a row's operand"),
        };
    }
    // The row's nodes the split reductions' operands need, in order.
    let reductions: Vec<&Node> = rows
        .nodes()
        .iter()
        .filter(|n| split.contains(&n.output))
        .collect();
    let mut needed = vec![false; rows.types.len()];
    let mut stack: Vec<Var> = reductions.iter().map(|r| r.inputs[0]).collect();
    let producer: std::collections::HashMap<Var, &Node> =
        rows.nodes().iter().map(|n| (n.output, n)).collect();
    while let Some(v) = stack.pop() {
        if let Some(node) = producer
            .get(&v)
            .filter(|_| !std::mem::replace(&mut needed[v], true))
        {
            stack.extend(&node.inputs);
        }
    }
    for node in rows.nodes().iter().filter(|n| needed[n.output]) {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the row");
    }
    let mut outputs = vec![product];
    for r in reductions {
        let x = map[r.inputs[0]];
        let n = *out.type_of(x).shape.last().expect("a row");
        let count = out.type_of(x).numel() / n;
        let new_sizes = vec![count, n / tile, *tile];
        let blocks = out
            .apply(Primitive::Reshape { new_sizes }, &[x])
            .expect("the rows in tiles");
        outputs.push(
            out.apply(over(r, vec![2]), &[blocks])
                .expect("their partials"),
        );
    }
    out.set_outputs(&outputs).expect("the partials' values");
    out
}

/// The row's body with each split reduction read as an extra input (after the row's own) instead of
/// computed.
fn finished_body(row: &Node, split: &[Var]) -> Graph {
    let Primitive::Fusion { body: rows, .. } = &row.primitive else {
        unreachable!("a row fusion")
    };
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];
    for &v in rows.inputs() {
        map[v] = out.input(rows.type_of(v).clone());
    }
    for &v in split {
        map[v] = out.input(rows.type_of(v).clone());
    }
    for node in rows.nodes().iter().filter(|n| !split.contains(&n.output)) {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the row");
    }
    let outputs: Vec<Var> = rows.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the row's values");
    out
}

/// Splits each matmul-and-row pair whose rows the backend's `tile` gives a partial tile width (rows too
/// wide for one GEMM tile) and whose leading reductions a tile can take partially: a `dot_partials`
/// fusion writes the value the row reads and those reductions' partials over each tile of a row; the
/// partials are reduced; the row reads them instead of reducing its rows again, so the product is read
/// once less. With `through`, one elementwise op may sit between the dot and the row, as in `dot_rows`.
pub(crate) fn dot_partials(
    graph: &Graph,
    joins: Joins,
    tile: fn(usize) -> Option<usize>,
    through: bool,
) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut split_of: Vec<Option<Split>> = vec![None; nodes.len()];
    let mut taken = vec![false; nodes.len()];
    for (r, row) in nodes.iter().enumerate() {
        let Some(pair @ (d, link, _)) = joined(graph, &producer, (r, row), joins, through) else {
            continue;
        };
        let Primitive::Fusion { body: rows, .. } = &row.primitive else {
            continue;
        };
        let root = rows.type_of(rows.outputs()[0]);
        let n = root.shape.last().copied().unwrap_or(1);
        let reductions = row_reductions(rows);
        let split: Vec<Var> = tile_reductions(rows, &reductions)
            .unwrap_or_default()
            .into_iter()
            .map(|k| reductions[k])
            .filter(|r| rows.type_of(r.inputs[0]).shape == root.shape)
            .map(|r| r.output)
            .collect();
        let Some(width) = tile(n) else { continue };
        if split.is_empty() || taken[d] || link.is_some_and(|e| taken[e]) {
            continue;
        }
        taken[d] = true;
        link.into_iter().for_each(|e| taken[e] = true);
        split_of[r] = Some((pair, width, split));
    }
    if split_of.iter().all(Option::is_none) {
        return graph.clone();
    }
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        if taken[i] {
            continue;
        }
        out.set_origin(node);
        let Some(split @ (pair @ (d, link, at), _, reduced)) = &split_of[i] else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");
            out.set_label(map[node.output], node.label);
            continue;
        };
        let Primitive::Fusion {
            body: rows, label, ..
        } = &node.primitive
        else {
            unreachable!("a row fusion")
        };
        let read: Vec<Var> = reads(graph, *pair, node).iter().map(|&v| map[v]).collect();
        let parts = std::iter::once(&nodes[*d])
            .chain(link.map(|e| &nodes[e]))
            .map(|n| graph.label(n))
            .chain(["partials"]);
        let body = partials_body(graph, split, node);
        let types: Vec<_> = body
            .outputs()
            .iter()
            .map(|&v| body.type_of(v).clone())
            .collect();
        let fused = Primitive::Fusion {
            name: "dot_partials".into(),
            label: intern(parts.collect::<Vec<_>>().join(FUSION_SEPARATOR)),
            body,
        };
        let value = out.apply(fused, &read).expect("the row's operand");
        if let Some(v) = link.map(|e| nodes[e].output) {
            map[v] = value;
        }
        // Each split reduction: its partials, reduced, in the reduction's own shape.
        let mut totals = Vec::new();
        for (k, &v) in reduced.iter().enumerate() {
            let r = rows
                .nodes()
                .iter()
                .find(|n| n.output == v)
                .expect("a split reduction");
            let ty = types[k + 1].clone();
            let partials = out
                .apply(Primitive::FusionOutput { index: k + 1, ty }, &[value])
                .expect("the partials");
            let total = out
                .apply(over(r, vec![1]), &[partials])
                .expect("the partials' total");
            let new_sizes = rows.type_of(v).shape.clone();
            let total = out
                .apply(Primitive::Reshape { new_sizes }, &[total])
                .expect("the reduction's shape");
            totals.push(total);
        }
        let inputs: Vec<Var> = node
            .inputs
            .iter()
            .enumerate()
            .map(|(k, &v)| match k == *at {
                true => value,
                false => map[v],
            })
            .chain(totals)
            .collect();
        let row = Primitive::Fusion {
            name: "row".into(),
            label,
            body: finished_body(node, reduced),
        };
        map[node.output] = out.apply(row, &inputs).expect("a fusion typed as its row");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}
