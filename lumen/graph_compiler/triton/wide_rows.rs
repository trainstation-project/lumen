//! Wide matmul rows on Triton: a normalization `row` fusion whose rows a GEMM (alone or with its
//! epilogue) makes, too wide for one GEMM tile to hold them whole, gets its GEMM writing the row's first
//! reductions' partials over each `tile`-column block of a row as well (`dot_partials`); those are
//! reduced and the row reads them instead of reducing its rows again, so the product is read once less.

use super::TritonModule;
use crate::codegen::dot_general::triton::partial_tile;
use crate::codegen::kinds::gemm_dot;
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::common::diamonds::tile_reductions;
use crate::graph_compiler::elementwise;
use crate::graph_compiler::pass::CompilerPass;

/// Name of a GEMM writing a row's operand and its first reductions' partials.
const DOT_PARTIALS: &str = "dot_partials";

/// Last-axis reductions of a row body, in order.
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

/// The shape of the rows a row body reduces: its first last-axis reduction's operand's.
fn rows_shape(body: &Graph) -> Option<Vec<usize>> {
    let first = row_reductions(body).into_iter().next()?;
    Some(body.type_of(first.inputs[0]).shape.clone())
}

/// Whether row body `body` reads its input `at` as whole rows at each element's own index: of its
/// rows' shape, not an output, read only by elementwise ops and its rows' reductions.
fn joins(body: &Graph, at: usize) -> bool {
    let v = body.inputs()[at];

    rows_shape(body).is_some_and(|rows| body.type_of(v).shape == rows)
        && !body.outputs().contains(&v)
        && body
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&v))
            .all(|n| {
                elementwise(&n.primitive)
                    || matches!(
                        n.primitive,
                        Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
                    )
            })
}

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

/// The computation node `g` stands for: its body when a GEMM fusion whose plain dot reads the body's
/// inputs, the node alone when a plain dot, else `None`.
fn gemm(graph: &Graph, g: &Node) -> Option<Graph> {
    match &g.primitive {
        Primitive::DotGeneral { .. } if plain(graph, g) => {
            let mut body = Graph::new();

            let inputs: Vec<Var> = g
                .inputs
                .iter()
                .map(|&v| body.input(graph.type_of(v).clone()))
                .collect();

            let product = body.apply(g.primitive.clone(), &inputs).expect("the dot");
            body.set_outputs(&[product]).expect("its product");
            Some(body)
        }
        Primitive::Fusion { body, .. } => {
            let dot = gemm_dot(body)?;
            let reads = dot.inputs.iter().all(|v| body.inputs().contains(v));
            (plain(body, dot) && reads).then(|| body.clone())
        }
        _ => None,
    }
}

/// A wide row to split: the GEMM node feeding it and its body, the row's operand slot it feeds, the
/// partial tile width, and the row body's reductions taken partially.
struct Wide {
    producer: usize,
    gemm: Graph,
    at: usize,
    tile: usize,
    split: Vec<Var>,
}

/// The wide row `row` (node `r`) a GEMM feeds, if any.
fn wide(graph: &Graph, producer: &[Option<usize>], (r, row): (usize, &Node)) -> Option<Wide> {
    let Primitive::Fusion {
        name, body: rows, ..
    } = &row.primitive
    else {
        return None;
    };

    if name != "row" {
        return None;
    }

    let nodes = graph.nodes();

    // The GEMM's values may be read again only after the row (they are the `dot_partials` outputs):
    // its extra outputs' `FusionOutput` nodes stay, read after the row too.
    let after = |v: Var| {
        nodes
            .iter()
            .enumerate()
            .all(|(k, n)| k >= r || !n.inputs.contains(&v))
    };

    let only = |v: Var| {
        nodes.iter().enumerate().all(|(k, n)| {
            k >= r
                || !n.inputs.contains(&v)
                || matches!(n.primitive, Primitive::FusionOutput { .. }) && after(n.output)
        })
    };

    let rows_shape = rows_shape(rows)?;
    let n = *rows_shape.last()?;
    let tile = partial_tile(n)?;
    let reductions = row_reductions(rows);

    let split: Vec<Var> = tile_reductions(rows, &reductions)
        .unwrap_or_default()
        .into_iter()
        .map(|k| reductions[k])
        .filter(|r| rows.type_of(r.inputs[0]).shape == rows_shape)
        .map(|r| r.output)
        .collect();

    if split.is_empty() {
        return None;
    }

    row.inputs.iter().enumerate().find_map(|(at, &v)| {
        let g = producer[v]?;
        let gemm = gemm(graph, &nodes[g]).filter(|_| joins(rows, at) && only(v))?;

        Some(Wide {
            producer: g,
            gemm,
            at,
            tile,
            split: split.clone(),
        })
    })
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

/// The `dot_partials` body: the GEMM's reads, then the row's other operands, in; the value the row reads,
/// each split reduction's partials over `tile`-column blocks of each row (`[rows, n / tile]`), then the
/// GEMM's other outputs, out.
fn partials_body(wide: &Wide, rows: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut gemm: Vec<Var> = vec![usize::MAX; wide.gemm.types.len()];

    for &v in wide.gemm.inputs() {
        gemm[v] = out.input(wide.gemm.type_of(v).clone());
    }

    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];

    for (k, &v) in rows.inputs().iter().enumerate() {
        if k != wide.at {
            map[v] = out.input(rows.type_of(v).clone());
        }
    }

    for node in wide.gemm.nodes() {
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| gemm[v]).collect();

        gemm[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("a node of the GEMM");
    }

    let product = gemm[wide.gemm.outputs()[0]];
    map[rows.inputs()[wide.at]] = product;

    // The row's nodes the split reductions' operands need, in order.
    let reductions: Vec<&Node> = rows
        .nodes()
        .iter()
        .filter(|n| wide.split.contains(&n.output))
        .collect();

    let producer: std::collections::HashMap<Var, &Node> =
        rows.nodes().iter().map(|n| (n.output, n)).collect();

    let mut needed = vec![false; rows.types.len()];
    let mut stack: Vec<Var> = reductions.iter().map(|r| r.inputs[0]).collect();

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
        let new_sizes = vec![count, n / wide.tile, wide.tile];

        let blocks = out
            .apply(Primitive::Reshape { new_sizes }, &[x])
            .expect("the rows in tiles");

        outputs.push(
            out.apply(over(r, vec![2]), &[blocks])
                .expect("their partials"),
        );
    }

    outputs.extend(wide.gemm.outputs()[1..].iter().map(|&v| gemm[v]));
    out.set_outputs(&outputs).expect("the partials' values");
    out
}

/// The row's body with each `split` reduction read as an extra input (after the row's own) instead of
/// computed.
fn finished_body(rows: &Graph, split: &[Var]) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; rows.types.len()];

    for &v in rows.inputs().iter().chain(split) {
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

/// Splits each wide row a GEMM feeds into a `dot_partials` fusion, the reduction of its partials, and
/// the row reading them.
fn wide_rows(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];

    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }

    let mut wide_of: Vec<Option<Wide>> = (0..nodes.len()).map(|_| None).collect();
    let mut taken = vec![false; nodes.len()];

    for (r, row) in nodes.iter().enumerate() {
        if let Some(w) = wide(graph, &producer, (r, row)).filter(|w| !taken[w.producer]) {
            taken[w.producer] = true;
            wide_of[r] = Some(w);
        }
    }

    if wide_of.iter().all(Option::is_none) {
        return graph.clone();
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![usize::MAX; graph.types.len()];

    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }

    // A taken GEMM's extra outputs, emitted with its `dot_partials` past the split reductions' partials.
    let extra = |n: &Node| match n.primitive {
        Primitive::FusionOutput { .. } => producer[n.inputs[0]].filter(|&g| taken[g]),
        _ => None,
    };

    for (i, node) in nodes.iter().enumerate() {
        if taken[i] || extra(node).is_some() {
            continue;
        }

        out.set_origin(node);

        let (
            Some(w),
            Primitive::Fusion {
                body: rows, label, ..
            },
        ) = (&wide_of[i], &node.primitive)
        else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();

            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("a node of the graph");

            out.set_label(map[node.output], node.label);
            continue;
        };

        let g = &nodes[w.producer];

        let reads: Vec<Var> = g
            .inputs
            .iter()
            .chain(
                node.inputs
                    .iter()
                    .enumerate()
                    .filter(|&(k, _)| k != w.at)
                    .map(|(_, v)| v),
            )
            .map(|&v| map[v])
            .collect();

        let body = partials_body(w, rows);

        let types: Vec<_> = body
            .outputs()
            .iter()
            .map(|&v| body.type_of(v).clone())
            .collect();

        let parts = [graph.label(g), "partials"].join(FUSION_SEPARATOR);

        let fused = Primitive::Fusion {
            name: DOT_PARTIALS.into(),
            label: intern(parts),
            body,
        };

        let value = out.apply(fused, &reads).expect("the row's operand");
        map[g.output] = value;

        for n in nodes.iter().filter(|n| extra(n) == Some(w.producer)) {
            let Primitive::FusionOutput { index, ty } = &n.primitive else {
                unreachable!("an extra output")
            };

            let index = index + w.split.len();
            let ty = ty.clone();

            map[n.output] = out
                .apply(Primitive::FusionOutput { index, ty }, &[value])
                .expect("the GEMM's extra output");
        }

        // Each split reduction: its partials, reduced, in the reduction's own shape.
        let mut totals = Vec::new();

        for (k, &v) in w.split.iter().enumerate() {
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
            .map(|(k, &v)| if k == w.at { value } else { map[v] })
            .chain(totals)
            .collect();

        let row = Primitive::Fusion {
            name: "row".into(),
            label,
            body: finished_body(rows, &w.split),
        };

        map[node.output] = out.apply(row, &inputs).expect("a fusion typed as its row");
    }

    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// Compiler pass splitting wide GEMM-fed rows into `dot_partials`, their partials' reduction and the row.
pub(super) struct WideRows;

impl CompilerPass<TritonModule<'_>> for WideRows {
    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "wide_rows"
    }

    /// Splits every wide row a GEMM feeds.
    fn apply(&mut self, m: &mut TritonModule<'_>) -> Result<(), String> {
        m.graph = wide_rows(&m.graph);
        Ok(())
    }
}
