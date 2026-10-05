//! Dot merging (XLA: `DotMerger`): dots that share an operand, as `x @ w1`
//! and `x @ w3` share `x`, become one dot of it with their other operands
//! side by side, `x @ [w1 | w3]`, sliced back into each one's result: one
//! larger matmul instead of several small ones. Nothing concatenates them:
//! the other operands must be parameters not yet placed, which are then
//! placed side by side in one block ([`crate::Tensor::pack`]), read by the
//! merged dot as a new input.
//!
//! Dots merge if they have the same dimension numbers and accumulation
//! dtype (the same primitive: one kernel accumulates in one dtype) and the
//! same operand on the same side, and their other operands are distinct packable
//! parameters, read by that dot alone (or, as contiguous parts of their
//! block, read in place, by others too: an optimizer's update), differing
//! only in their last free dimension (the one they are side by side along). Only dots whose results
//! are read by primitives that fuse, or dots (and are not outputs), merge:
//! those read their slices in place, where any other reader would need a
//! copy of its own.
//!
//! A dot's operand may be computed from its parameter elementwise too (a
//! chain of elementwise primitives, each of it and splat constants alone:
//! mixed precision's bfloat16 copy of a float32 weight, a scaled weight),
//! the same chain for each dot of a group: the merged dot reads the chain
//! of their block, computed once (elementwise, the block's chain is each
//! one's side by side); any other reader of a dot's operand (a backward's
//! dot of the same bfloat16 weight) reads its part of it.
//!
//! A parameter the program assigns (donated: an optimizer's update, its new
//! value written over it in place) merges only as a contiguous part of its
//! block, side by side along its first dimension (past any of size 1): as
//! PyTorch's fused QKV weights, `[3D, D]` blocks of `nn.Linear`'s `[out,
//! in]` weights, each a slice an optimizer updates in place.

use super::fusion::{elementwise, fusible};
use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};

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
    /// The chain their operands are computed from their parameters by
    /// (none: the parameters themselves).
    chain: Vec<Link>,
    nodes: Vec<usize>,
    /// Their parameters, each once.
    weights: Vec<Var>,
}

/// One elementwise primitive of a chain from a parameter to a dot's
/// operand ([`merge_dots`]): of the value before it alone, or of it and a
/// splat constant (the operand position of the scalar `full` broadcast).
#[derive(Debug, Clone, PartialEq)]
struct Link {
    primitive: Primitive,
    splat: Option<(usize, Primitive)>,
}

/// `graph` with each group of dots that share an operand merged into one,
/// and the inputs it adds: each the block of these inputs (positions,
/// among those `packable`; those `donated` contiguous in it) side by side
/// along a dimension.
pub(crate) fn merge_dots(
    graph: &Graph,
    packable: &[bool],
    donated: &[bool],
) -> (Graph, Vec<(Vec<usize>, usize)>) {
    let nodes = graph.nodes();
    // Each packable input read by one node alone: its position. A
    // concatenate of packable inputs alone does not count: if they merge,
    // in its order and dimension, it is their block (a backward's, of the
    // weights its dot's merged dots read, `lumen/autograd/dots.py`).
    let is_packable = |v: Var| {
        let i = graph.inputs().iter().position(|&i| i == v);
        i.is_some_and(|i| packable.get(i) == Some(&true))
    };
    let of_inputs = |node: &Node| {
        matches!(node.primitive, Primitive::Concatenate { .. })
            && node.inputs.iter().all(|&v| is_packable(v))
    };
    let mut reads = vec![0; graph.types.len()];
    for &v in nodes
        .iter()
        .filter(|node| !of_inputs(node))
        .flat_map(|node| &node.inputs)
        .chain(graph.outputs())
    {
        reads[v] += 1;
    }
    // Each packable input: its position, and whether one node alone reads
    // it (else it merges only as a contiguous part of its block).
    let mut position: Vec<Option<usize>> = vec![None; graph.types.len()];
    let mut alone = vec![false; graph.types.len()];
    for (i, &v) in graph.inputs().iter().enumerate() {
        if packable.get(i) == Some(&true) && reads[v] >= 1 {
            position[v] = Some(i);
            alone[v] = reads[v] == 1;
        }
    }
    // The parameter each dot operand is, or is computed from by a chain of
    // elementwise primitives (each value before the operand read by the
    // next alone), and the chain.
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
    // A scalar `full` broadcast: the `full`.
    let splat = |v: Var| -> Option<Primitive> {
        let b = &nodes[producer[v]?];
        let Primitive::BroadcastInDim { .. } = b.primitive else {
            return None;
        };
        let f = &nodes[producer[b.inputs[0]]?];
        matches!(&f.primitive, Primitive::Full { shape, .. } if shape.is_empty())
            .then(|| f.primitive.clone())
    };
    let weight_of = |v: Var| -> Option<(Var, Vec<Link>)> {
        let (mut chain, mut x) = (Vec::new(), v);
        while position[x].is_none() {
            if x != v && reads[x] != 1 {
                return None;
            }
            let node = &nodes[producer[x]?];
            if !elementwise(&node.primitive) {
                return None;
            }
            let (next, constant) = match *node.inputs.as_slice() {
                [a] => (a, None),
                [a, b] => match (splat(a), splat(b)) {
                    (None, Some(f)) => (a, Some((1, f))),
                    (Some(f), None) => (b, Some((0, f))),
                    _ => return None,
                },
                _ => return None,
            };
            chain.push(Link {
                primitive: node.primitive.clone(),
                splat: constant,
            });
            x = next;
        }
        chain.reverse();
        Some((x, chain))
    };
    // Whether each value is read only in place: by fusible primitives, or
    // by dots (which read a slice as a strided view of it).
    let mut read_in_place = vec![true; graph.types.len()];
    for &v in graph.outputs() {
        read_in_place[v] = false;
    }
    let in_place = |node: &&Node| {
        fusible(graph, node) || matches!(node.primitive, Primitive::DotGeneral { .. })
    };
    for node in nodes.iter().filter(|node| !in_place(node)) {
        node.inputs.iter().for_each(|&v| read_in_place[v] = false);
    }
    // Each dot joins the first group it can, else starts one on each side.
    let mut groups: Vec<Group> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        if !matches!(node.primitive, Primitive::DotGeneral { .. }) || !read_in_place[node.output] {
            continue;
        }
        let mut candidates = Vec::new();
        for side in 0..2 {
            let Some((w, chain)) = weight_of(node.inputs[1 - side]) else {
                continue;
            };
            let other = graph.type_of(node.inputs[1 - side]);
            let Some(dimension) = concat_dimension(&node.primitive, side, other.shape.len()) else {
                continue;
            };
            // A donated parameter, or one read elsewhere too, only as a
            // contiguous part of the block.
            let i = position[w].expect("a packable input");
            let strided = other.shape[..dimension].iter().any(|&n| n != 1);
            if strided && (donated.get(i) == Some(&true) || !alone[w]) {
                continue;
            }
            let mut key = other.clone();
            key.shape[dimension] = 0;
            candidates.push((side, key, dimension, chain, w));
        }
        let joins = groups.iter().position(|g| {
            candidates.iter().any(|(side, key, dimension, chain, w)| {
                g.shared == node.inputs[*side]
                    && g.side == *side
                    && g.primitive == node.primitive
                    && g.other == *key
                    && g.dimension == *dimension
                    && g.chain == *chain
                    && !g.weights.contains(w)
            })
        });
        match joins {
            Some(g) => {
                let w = candidates
                    .iter()
                    .find(|(side, ..)| groups[g].side == *side)
                    .expect("the side it joins on")
                    .4;
                groups[g].nodes.push(i);
                groups[g].weights.push(w);
            }
            None => groups.extend(candidates.into_iter().map(
                |(side, other, dimension, chain, w)| Group {
                    shared: node.inputs[side],
                    side,
                    primitive: node.primitive.clone(),
                    other,
                    dimension,
                    chain,
                    nodes: vec![i],
                    weights: vec![w],
                },
            )),
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

    let mut packs = Vec::new();
    // Each merged group's block: its operands in order, its dimension, its
    // input.
    let mut blocks: Vec<(Vec<Var>, usize, Var)> = Vec::new();
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    for (i, node) in nodes.iter().enumerate() {
        out.set_scope(node.scope);
        if let Some(k) = first_of[i] {
            let g = &groups[k];
            // The parameters (behind their chains).
            let others: Vec<Var> = g
                .nodes
                .iter()
                .map(|&m| {
                    weight_of(nodes[m].inputs[1 - g.side])
                        .expect("a parameter")
                        .0
                })
                .collect();
            let types: Vec<&TensorType> = others.iter().map(|&v| graph.type_of(v)).collect();
            let block = Primitive::Concatenate {
                dimension: g.dimension,
            }
            .infer(&types)
            .expect("operands equal but in the dimension");
            packs.push((
                others.iter().filter_map(|&v| position[v]).collect(),
                g.dimension,
            ));
            let mut operands = [map[g.shared]; 2];
            let block = out.input(block);
            blocks.push((others.clone(), g.dimension, block));
            // Their chain: once, of the block.
            let mut value = block;
            for link in &g.chain {
                let mut ins = vec![value];
                if let Some((k, full)) = &link.splat {
                    let s = out.apply(full.clone(), &[]).expect("a scalar");
                    let shape = out.type_of(value).shape.clone();
                    let b = Primitive::BroadcastInDim {
                        shape,
                        broadcast_dimensions: Vec::new(),
                    };
                    ins.insert(*k, out.apply(b, &[s]).expect("a splat"));
                }
                value = out
                    .apply(link.primitive.clone(), &ins)
                    .expect("the chain, of the block");
            }
            operands[1 - g.side] = value;
            // Each dot's operand, for its other readers: its part of the
            // block's chain.
            if !g.chain.is_empty() {
                let shape = out.type_of(value).shape.clone();
                let mut start = 0;
                for &m in &g.nodes {
                    let v = nodes[m].inputs[1 - g.side];
                    let n = graph.type_of(v).shape[g.dimension];
                    let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                    (starts[g.dimension], limits[g.dimension]) = (start, start + n);
                    let part = Primitive::Slice {
                        start_indices: starts,
                        limit_indices: limits,
                    };
                    map[v] = out.apply(part, &[value]).expect("within the block");
                    start += n;
                }
            }
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
        // A concatenate of a merged group's operands, as they are side by
        // side: their block.
        if let Primitive::Concatenate { dimension } = node.primitive
            && let Some((_, _, block)) = blocks
                .iter()
                .find(|(others, d, _)| *others == node.inputs && *d == dimension)
        {
            map[node.output] = *block;
            continue;
        }
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the node's own operands");
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    (out, packs)
}

/// The batch and contracting dimensions of a dot's operand on `side`.
fn used_dims(p: &Primitive, side: usize) -> Vec<usize> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
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
