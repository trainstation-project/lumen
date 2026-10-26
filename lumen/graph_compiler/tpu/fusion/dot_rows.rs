//! Fuses a matmul into the normalization `row` fusion that consumes its whole output rows.
use super::super::TpuModule;
use crate::codegen::tpu::row::{Kind, kinds};
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::pass::CompilerPass;

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

/// Finds the plain dot feeding `row` as a full-row operand, returning the dot's node and operand slot.
fn joined(graph: &Graph, producer: &[Option<usize>], row: &Node) -> Option<(usize, usize)> {
    let Primitive::Fusion { name, body, .. } = &row.primitive else {
        return None;
    };
    if name != "row" {
        return None;
    }
    let kinds = kinds(body)?;
    let nodes = graph.nodes();
    // The dot's result must be read whole by the row kernel, by nothing else, and not escape the graph.
    row.inputs.iter().enumerate().find_map(|(at, &v)| {
        let d = producer[v]?;
        let only = nodes
            .iter()
            .filter(|n| n.inputs.contains(&v))
            .all(|n| std::ptr::eq(n, row));
        (kinds[body.inputs()[at]] == Kind::Full
            && plain(graph, &nodes[d])
            && only
            && !graph.outputs().contains(&v))
        .then_some((d, at))
    })
}

/// Builds the `dot_row` body: the dot's operands and the row's other reads in, the row's values out.
fn body(graph: &Graph, dot: &Node, row: &Node, at: usize) -> Graph {
    let Primitive::Fusion { body: rows, .. } = &row.primitive else {
        unreachable!("a row fusion")
    };
    let mut out = Graph::new();
    let lhs = out.input(graph.type_of(dot.inputs[0]).clone());
    let rhs = out.input(graph.type_of(dot.inputs[1]).clone());
    let others: Vec<Var> = row
        .inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != at)
        .map(|(_, &v)| out.input(graph.type_of(v).clone()))
        .collect();
    let product = out
        .apply(dot.primitive.clone(), &[lhs, rhs])
        .expect("the dot");
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
    let outputs: Vec<Var> = rows.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("the row's values");
    out
}

/// Replaces each matched row fusion and its feeding dot with one `dot_row` fusion.
pub(super) fn dot_rows(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    let mut dot_of: Vec<Option<(usize, usize)>> = vec![None; nodes.len()];
    let mut taken = vec![false; nodes.len()];
    for (r, row) in nodes.iter().enumerate() {
        if let Some((d, at)) = joined(graph, &producer, row)
            && !taken[d]
        {
            taken[d] = true;
            dot_of[r] = Some((d, at));
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
        if let Some((d, at)) = dot_of[i] {
            let dot = &nodes[d];
            // Reads are the dot's lhs and rhs followed by the row's remaining operands.
            let reads: Vec<Var> = [dot.inputs[0], dot.inputs[1]]
                .into_iter()
                .chain(
                    node.inputs
                        .iter()
                        .enumerate()
                        .filter(|&(k, _)| k != at)
                        .map(|(_, &v)| v),
                )
                .map(|v| map[v])
                .collect();
            let label = [graph.label(dot), graph.label(node)].join(FUSION_SEPARATOR);
            let fused = Primitive::Fusion {
                name: "dot_row".into(),
                label: intern(label),
                body: body(graph, dot, node, at),
            };
            map[node.output] = out.apply(fused, &reads).expect("a fusion typed as its row");
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

/// Fuses dots into the row fusions that read them when contraction epilogues are enabled.
pub(in crate::graph_compiler::tpu) struct DotRows;

impl CompilerPass<TpuModule<'_>> for DotRows {
    fn name(&self) -> &'static str {
        "dot_rows"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        if m.options.config.tpu.contraction_epilogues {
            m.graph = dot_rows(&m.graph);
        }
        Ok(())
    }
}
