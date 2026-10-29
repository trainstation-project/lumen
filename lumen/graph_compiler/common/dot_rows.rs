//! Fuses a matmul into the normalization `row` fusion that consumes its whole output rows, as one
//! `dot_row` fusion: the row computed from the product without it reaching memory. With `through`, one
//! elementwise op may sit between them (a residual added to the product), its value written too where
//! it is read again.

use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
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
