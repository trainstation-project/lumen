//! Dot merging (XLA: `DotMerger`): dots that share an operand, as `x @ w1`
//! and `x @ w3` share `x`, become one dot of it with their other operands
//! concatenated, `x @ [w1 | w3]`, sliced back into each one's result: one
//! larger matmul instead of several small ones. The concatenation copies
//! the operands on every run, so as in XLA only while the concatenated
//! operand and the result fit in [`MAX_BYTES`].
//!
//! Dots merge if they have the same dimension numbers and the same operand
//! on the same side, and their other operands differ only in their last
//! free dimension (the one concatenated). Only dots whose results are read
//! by primitives that fuse (and are not outputs) merge: those read their
//! slices in place, where any other reader would need a copy of its own. The merged dot replaces the
//! group's first, so each other operand must be defined before it, which
//! also keeps any dot from depending on another of its group.

use super::fusion::fusible;
use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Primitive, TensorType, Var};

/// XLA's default `xla_gpu_dot_merger_threshold_mb`.
const MAX_BYTES: usize = 32 << 20;

/// Dots that can merge: their nodes, in graph order.
struct Group {
    /// The operand they share, and its side (0 for lhs, 1 for rhs).
    shared: Var,
    side: usize,
    primitive: Primitive,
    /// Their other operands' type, with the concatenated dimension 0.
    other: TensorType,
    /// The concatenated dimension of those operands.
    dimension: usize,
    nodes: Vec<usize>,
    /// The bytes of the concatenated operand and the merged result.
    bytes: usize,
}

/// `graph` with each group of dots that share an operand merged into one.
pub(crate) fn merge_dots(graph: &Graph) -> Graph {
    let nodes = graph.nodes();
    let mut defined_at = vec![0; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        defined_at[node.output] = i + 1;
    }
    // Whether each value is read only by fusible primitives, in place.
    let mut read_in_place = vec![true; graph.types.len()];
    for &v in graph.outputs() {
        read_in_place[v] = false;
    }
    for node in nodes.iter().filter(|node| !fusible(graph, node)) {
        node.inputs.iter().for_each(|&v| read_in_place[v] = false);
    }
    let bytes = |t: &TensorType| t.numel() * t.dtype.size_of();
    // Each dot joins the first group it can, else starts one on each side.
    let mut groups: Vec<Group> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.primitive, Primitive::DotGeneral { .. }) || !read_in_place[node.output] {
            continue;
        }
        let out_bytes = bytes(graph.type_of(node.output));
        let mut candidates = Vec::new();
        for side in 0..2 {
            let other = graph.type_of(node.inputs[1 - side]);
            let Some(dimension) = concat_dimension(&node.primitive, side, other.shape.len()) else {
                continue;
            };
            let mut key = other.clone();
            key.shape[dimension] = 0;
            candidates.push((side, key, dimension, bytes(other) + out_bytes));
        }
        let joins = groups.iter().position(|g| {
            candidates.iter().any(|(side, key, dimension, added)| {
                g.shared == node.inputs[*side]
                    && g.side == *side
                    && g.primitive == node.primitive
                    && g.other == *key
                    && g.dimension == *dimension
                    && g.bytes + added <= MAX_BYTES
                    && defined_at[node.inputs[1 - side]] <= g.nodes[0]
            })
        });
        match joins {
            Some(g) => {
                let g = &mut groups[g];
                g.nodes.push(i);
                g.bytes += bytes(graph.type_of(node.inputs[1 - g.side])) + out_bytes;
            }
            None => groups.extend(
                candidates
                    .into_iter()
                    .map(|(side, other, dimension, bytes)| Group {
                        shared: node.inputs[side],
                        side,
                        primitive: node.primitive.clone(),
                        other,
                        dimension,
                        nodes: vec![i],
                        bytes,
                    }),
            ),
        }
    }
    // A dot in two groups of two or more merges in the first; the rest of
    // the second, if still two or more, merge at their first.
    let mut merged = vec![false; nodes.len()];
    let mut first_of: Vec<Option<usize>> = vec![None; nodes.len()];
    for (k, g) in groups.iter_mut().enumerate() {
        g.nodes.retain(|&i| !merged[i]);
        if g.nodes.len() > 1 {
            g.nodes.iter().for_each(|&i| merged[i] = true);
            first_of[g.nodes[0]] = Some(k);
        }
    }

    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        if let Some(k) = first_of[i] {
            let g = &groups[k];
            let others: Vec<Var> = g
                .nodes
                .iter()
                .map(|&m| map[nodes[m].inputs[1 - g.side]])
                .collect();
            let concatenated = out
                .apply(
                    Primitive::Concatenate {
                        dimension: g.dimension,
                    },
                    &others,
                )
                .expect("operands equal but in the dimension");
            let mut operands = [map[g.shared]; 2];
            operands[1 - g.side] = concatenated;
            let result = out
                .apply(g.primitive.clone(), &operands)
                .expect("the dots' dimension numbers");
            // The concatenated dimension is its operand's last free one, so
            // the result's last of that operand's: its last for an rhs, else
            // the one before the rhs's free dimensions.
            let rank = out.type_of(result).shape.len();
            let dimension = match g.side {
                0 => rank - 1,
                _ => {
                    let rhs = graph.type_of(node.inputs[1]).shape.len();
                    rank - 1 - free_dims(rhs, &used_dims(&g.primitive, 1)).count()
                }
            };
            let shape = out.type_of(result).shape.clone();
            let mut start = 0;
            for &m in &g.nodes {
                let size = graph.type_of(nodes[m].output).shape[dimension];
                let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                (starts[dimension], limits[dimension]) = (start, start + size);
                let slice = Primitive::Slice {
                    start_indices: starts,
                    limit_indices: limits,
                };
                map[nodes[m].output] = out.apply(slice, &[result]).expect("within the result");
                start += size;
            }
            continue;
        }
        if merged[i] {
            continue;
        }
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the node's own operands");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    out
}

/// The batch and contracting dimensions of a dot's operand on `side`.
fn used_dims(p: &Primitive, side: usize) -> Vec<usize> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
    } = p
    else {
        unreachable!("a dot")
    };
    match side {
        0 => [lhs_batch.as_slice(), lhs_contracting].concat(),
        _ => [rhs_batch.as_slice(), rhs_contracting].concat(),
    }
}

/// The dimension the operands opposite `side` concatenate along: the last
/// free one of the operand, if it has one.
fn concat_dimension(p: &Primitive, side: usize, rank: usize) -> Option<usize> {
    free_dims(rank, &used_dims(p, 1 - side)).last()
}
