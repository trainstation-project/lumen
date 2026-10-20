use std::collections::HashMap;

use super::MpsModule;
use super::fusion::{elementwise, fusible};
use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::combined;
use crate::graph_compiler::pass::CompilerPass;

struct Group {
    shared: Var,
    side: usize,
    primitive: Primitive,
    other: TensorType,
    dimension: usize,
    chain: Vec<Link>,
    nodes: Vec<usize>,
    weights: Vec<Var>,
}

#[derive(Debug, Clone, PartialEq)]
struct Link {
    primitive: Primitive,
    splat: Option<(usize, Primitive)>,
}

type Packs = Vec<(Vec<usize>, usize)>;

pub(super) fn merge_dots(
    graph: &Graph,
    packable: &[bool],
    donated: &[bool],
) -> (Graph, Packs, Vec<usize>) {
    let nodes = graph.nodes();
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
    let mut position: Vec<Option<usize>> = vec![None; graph.types.len()];
    let mut alone = vec![false; graph.types.len()];
    for (i, &v) in graph.inputs().iter().enumerate() {
        if packable.get(i) == Some(&true) && reads[v] >= 1 {
            position[v] = Some(i);
            alone[v] = reads[v] == 1;
        }
    }
    let mut producer: Vec<Option<usize>> = vec![None; graph.types.len()];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
    }
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
    let mut merged = vec![false; nodes.len()];
    let mut first_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut packed = vec![false; graph.types.len()];
    let mut reuses: Vec<Option<usize>> = vec![None; groups.len()];
    let mut interleave: Vec<usize> = vec![1; groups.len()];
    let mut cotangents: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut pairs_of: Vec<Option<(usize, usize, usize)>> = vec![None; nodes.len()];
    let weight =
        |side: usize, i: usize| weight_of(nodes[i].inputs[1 - side]).expect("a parameter").0;
    for k in 0..groups.len() {
        let side = groups[k].side;
        groups[k].nodes.retain(|&i| !merged[i]);
        let weights: Vec<Var> = groups[k].nodes.iter().map(|&i| weight(side, i)).collect();
        let earlier = (0..k).find(|&e| {
            let (g, b) = (&groups[k], &groups[e]);
            reuses[e].is_none()
                && b.nodes.len() > 1
                && b.nodes
                    .iter()
                    .map(|&i| weight(b.side, i))
                    .eq(weights.iter().copied())
                && (b.side, &b.chain, b.dimension) == (g.side, &g.chain, g.dimension)
        });
        if let Some(e) = earlier {
            groups[k].nodes.iter().for_each(|&i| merged[i] = true);
            first_of[groups[k].nodes[0]] = Some(k);
            reuses[k] = Some(e);
            interleave[k] = interleave[e];
            continue;
        }
        let g = &mut groups[k];
        g.nodes.retain(|&i| !packed[weight(side, i)]);
        if g.nodes.len() > 1 {
            for &i in &g.nodes {
                merged[i] = true;
                packed[weight(side, i)] = true;
            }
            first_of[g.nodes[0]] = Some(k);
            if let Some(rewrites) = interleaves(graph, g, &position, &alone, &reads, donated) {
                interleave[k] = g.nodes.len();
                for &c in &rewrites.cotangents {
                    cotangents[c] = Some(interleave[k]);
                }
                for &(s, j, axis) in &rewrites.slices {
                    pairs_of[s] = Some((j, interleave[k], axis));
                }
            }
        }
    }

    let mut packs = Vec::new();
    let mut interleaved = Vec::new();
    let mut tupled: HashMap<(Var, usize), Var> = HashMap::new();
    let mut blocks: Vec<(Vec<Var>, usize, Var)> = Vec::new();
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut parted = vec![false; graph.types.len()];
    let mut chains: Vec<Option<Var>> = vec![None; groups.len()];
    let mut parts: Vec<(Vec<Var>, usize, Var)> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        if parted[node.output] {
            continue;
        }
        if let Some(k) = first_of[i] {
            let g = &groups[k];
            out.set_origins(&g.nodes.iter().map(|&m| &nodes[m]).collect::<Vec<_>>());
            let mut operands = [map[g.shared]; 2];
            let value = match reuses[k] {
                Some(e) => chains[e].expect("an earlier group's block"),
                None => {
                    let others: Vec<Var> = g
                        .nodes
                        .iter()
                        .map(|&m| {
                            weight_of(nodes[m].inputs[1 - g.side])
                                .expect("a parameter")
                                .0
                        })
                        .collect();
                    let types: Vec<&TensorType> =
                        others.iter().map(|&v| graph.type_of(v)).collect();
                    let block = Primitive::Concatenate {
                        dimension: g.dimension,
                        in_place: false,
                    }
                    .infer(&types)
                    .expect("operands equal but in the dimension");
                    packs.push((
                        others.iter().filter_map(|&v| position[v]).collect(),
                        g.dimension,
                    ));
                    interleaved.push(interleave[k]);
                    let block = out.input(block);
                    blocks.push((others.clone(), g.dimension, block));
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
                    chains[k] = Some(value);
                    value
                }
            };
            operands[1 - g.side] = value;
            parts.push((
                g.nodes
                    .iter()
                    .map(|&m| nodes[m].inputs[1 - g.side])
                    .collect(),
                g.dimension,
                value,
            ));
            if interleave[k] == 1 && !g.chain.is_empty() {
                let shape = out.type_of(value).shape.clone();
                let mut start = 0;
                for &m in &g.nodes {
                    let v = nodes[m].inputs[1 - g.side];
                    let n = graph.type_of(v).shape[g.dimension];
                    if parted[v] {
                        start += n;
                        continue;
                    }
                    let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                    (starts[g.dimension], limits[g.dimension]) = (start, start + n);
                    let part = Primitive::Slice {
                        start_indices: starts,
                        limit_indices: limits,
                        view: false,
                    };
                    map[v] = out.apply(part, &[value]).expect("within the block");
                    parted[v] = true;
                    start += n;
                }
            }
            let result = out
                .apply(g.primitive.clone(), &operands)
                .expect("the dots' dimension numbers");
            let rank = out.type_of(result).shape.len();
            let dimension = match g.side {
                0 => rank - 1,
                _ => {
                    let rhs = graph.type_of(node.inputs[1]).shape.len();
                    rank - 1 - free_dims(rhs, &used_dims(&g.primitive, 1)).count()
                }
            };
            let shape = out.type_of(result).shape.clone();
            if interleave[k] > 1 {
                for (j, &m) in g.nodes.iter().enumerate() {
                    out.set_origin(&nodes[m]);
                    let member = &graph.type_of(nodes[m].output).shape;
                    map[nodes[m].output] = pairs(
                        &mut out,
                        &mut tupled,
                        result,
                        (dimension, interleave[k], j),
                        member,
                    );
                }
                continue;
            }
            let mut start = 0;
            for &m in &g.nodes {
                let size = graph.type_of(nodes[m].output).shape[dimension];
                let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                (starts[dimension], limits[dimension]) = (start, start + size);
                let slice = Primitive::Slice {
                    start_indices: starts,
                    limit_indices: limits,
                    view: false,
                };
                out.set_origin(&nodes[m]);
                map[nodes[m].output] = out.apply(slice, &[result]).expect("within the result");
                start += size;
            }
            continue;
        }
        if merged[i] {
            continue;
        }
        if let (
            Some(kk),
            Primitive::Concatenate {
                dimension: axis, ..
            },
        ) = (cotangents[i], &node.primitive)
        {
            let shape = graph.type_of(node.output).shape.clone();
            let tuples: Vec<Var> = node
                .inputs
                .iter()
                .map(|&v| {
                    let mut one = graph.type_of(v).shape.clone();
                    one.insert(axis + 1, 1);
                    let reshape = Primitive::Reshape { new_sizes: one };
                    out.apply(reshape, &[map[v]]).expect("the same elements")
                })
                .collect();
            let tuples = out
                .apply(
                    Primitive::Concatenate {
                        dimension: axis + 1,
                        in_place: false,
                    },
                    &tuples,
                )
                .expect("parts of one shape");
            debug_assert_eq!(out.type_of(tuples).shape[axis + 1], kk);
            let back = Primitive::Reshape { new_sizes: shape };
            map[node.output] = out.apply(back, &[tuples]).expect("the same elements");
            continue;
        }
        if let Some((j, kk, axis)) = pairs_of[i] {
            let shape = &graph.type_of(node.output).shape;
            map[node.output] = pairs(
                &mut out,
                &mut tupled,
                map[node.inputs[0]],
                (axis, kk, j),
                shape,
            );
            continue;
        }
        if let Primitive::Concatenate { dimension, .. } = node.primitive
            && let Some(&(_, _, block)) = blocks
                .iter()
                .chain(&parts)
                .find(|(others, d, _)| *others == node.inputs && *d == dimension)
        {
            map[node.output] = block;
            continue;
        }
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the node's own operands");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs).expect("values of the graph");
    (out, packs, interleaved)
}

fn pairs(
    out: &mut Graph,
    tupled: &mut HashMap<(Var, usize), Var>,
    value: Var,
    (axis, k, j): (usize, usize, usize),
    shape: &[usize],
) -> Var {
    let mut tuples = out.type_of(value).shape.clone();
    tuples.splice(axis..=axis, [tuples[axis] / k, k]);
    let reshaped = *tupled.entry((value, axis)).or_insert_with(|| {
        let reshape = Primitive::Reshape {
            new_sizes: tuples.clone(),
        };
        out.apply(reshape, &[value]).expect("the same elements")
    });
    let (mut starts, mut limits) = (vec![0; tuples.len()], tuples);
    (starts[axis + 1], limits[axis + 1]) = (j, j + 1);
    let slice = Primitive::Slice {
        start_indices: starts,
        limit_indices: limits,
        view: false,
    };
    let one = out.apply(slice, &[reshaped]).expect("within the tuples");
    let back = Primitive::Reshape {
        new_sizes: shape.to_vec(),
    };
    out.apply(back, &[one]).expect("the member's elements")
}

#[derive(Default)]
struct Interleaving {
    cotangents: Vec<usize>,
    slices: Vec<(usize, usize, usize)>,
}

fn interleaves(
    graph: &Graph,
    g: &Group,
    position: &[Option<usize>],
    alone: &[bool],
    reads: &[usize],
    donated: &[bool],
) -> Option<Interleaving> {
    let nodes = graph.nodes();
    let operands: Vec<Var> = g
        .nodes
        .iter()
        .map(|&i| nodes[i].inputs[1 - g.side])
        .collect();
    let results: Vec<Var> = g.nodes.iter().map(|&i| nodes[i].output).collect();
    let size = |w: Var| graph.type_of(w).shape[g.dimension];
    let sets = combined(graph, &results);
    let fits = g.nodes.len() == 2
        && g.side == 0
        && sets.iter().all(|&s| s == sets[0])
        && g.weights.iter().all(|&w| {
            let i = position[w].expect("a packable input");
            alone[w] && donated.get(i) != Some(&true) && size(w) == size(g.weights[0])
        })
        && operands.iter().all(|&v| reads[v] == 1);
    match fits {
        true => backward(graph, g, &operands),
        false => None,
    }
}

fn backward(graph: &Graph, g: &Group, operands: &[Var]) -> Option<Interleaving> {
    let nodes = graph.nodes();
    let k = g.nodes.len();
    let outputs = graph.outputs();
    let readers = |v: Var| {
        nodes
            .iter()
            .enumerate()
            .filter(move |(_, n)| n.inputs.contains(&v))
    };
    let produced = |v: Var| nodes.iter().position(|n| n.output == v);
    let concatenate_of_k = |n: &Node, axis: usize| {
        matches!(n.primitive, Primitive::Concatenate { dimension, .. } if dimension == axis)
            && n.inputs.len() == k
            && n.inputs
                .iter()
                .all(|&v| graph.type_of(v).shape[axis] == graph.type_of(n.inputs[0]).shape[axis])
    };
    let mut found = Interleaving::default();
    let mut contractions = Vec::new();
    for b in nodes.iter().filter(|n| {
        matches!(n.primitive, Primitive::Concatenate { dimension, .. } if dimension == g.dimension)
            && (n.inputs == g.weights || n.inputs == operands)
    }) {
        if outputs.contains(&b.output) {
            return None;
        }
        for (r, reader) in readers(b.output) {
            let Primitive::DotGeneral { .. } = reader.primitive else {
                return None;
            };
            let side = reader.inputs.iter().position(|&v| v == b.output)?;
            let (mine, theirs) = (
                contracting(&reader.primitive, side),
                contracting(&reader.primitive, 1 - side),
            );
            let c = mine.iter().position(|&d| d == g.dimension)?;
            let cotangent = &nodes[produced(reader.inputs[1 - side])?];
            if !concatenate_of_k(cotangent, theirs[c]) {
                return None;
            }
            let at = produced(cotangent.output)?;
            if !found.cotangents.contains(&at) {
                found.cotangents.push(at);
            }
            contractions.push(r);
        }
    }
    for &at in &found.cotangents {
        let cotangent = &nodes[at];
        let Primitive::Concatenate {
            dimension: axis, ..
        } = cotangent.primitive
        else {
            unreachable!("a concatenate")
        };
        if outputs.contains(&cotangent.output) {
            return None;
        }
        for (r, reader) in readers(cotangent.output) {
            if contractions.contains(&r) {
                continue;
            }
            let Primitive::DotGeneral { lhs_batch, .. } = &reader.primitive else {
                return None;
            };
            let side = reader.inputs.iter().position(|&v| v == cotangent.output)?;
            let rank = graph.type_of(cotangent.output).shape.len();
            let free: Vec<usize> = free_dims(rank, &used_dims(&reader.primitive, side)).collect();
            let pos = free.iter().position(|&d| d == axis)?;
            let before = match side {
                0 => 0,
                _ => {
                    let lhs = graph.type_of(reader.inputs[0]).shape.len();
                    free_dims(lhs, &used_dims(&reader.primitive, 0)).count()
                }
            };
            let result_axis = lhs_batch.len() + before + pos;
            let result = graph.type_of(reader.output);
            let n = result.shape[result_axis] / k;
            if outputs.contains(&reader.output) {
                return None;
            }
            let mut members = vec![false; k];
            for (s, slice) in readers(reader.output) {
                let Primitive::Slice {
                    start_indices,
                    limit_indices,
                    ..
                } = &slice.primitive
                else {
                    return None;
                };
                let whole = |d: usize| start_indices[d] == 0 && limit_indices[d] == result.shape[d];
                let j = start_indices[result_axis] / n.max(1);
                let part = start_indices[result_axis] == j * n
                    && limit_indices[result_axis] == (j + 1) * n
                    && (0..result.shape.len()).all(|d| d == result_axis || whole(d));
                if !part || j >= k || members[j] {
                    return None;
                }
                members[j] = true;
                found.slices.push((s, j, result_axis));
            }
        }
    }
    Some(found)
}

fn contracting(p: &Primitive, side: usize) -> Vec<usize> {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        ..
    } = p
    else {
        unreachable!("a dot")
    };
    match side {
        0 => lhs_contracting.clone(),
        _ => rhs_contracting.clone(),
    }
}

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

fn concat_dimension(p: &Primitive, side: usize, rank: usize) -> Option<usize> {
    free_dims(rank, &used_dims(p, 1 - side)).last()
}

pub(super) struct MergeDots;

impl<'a> CompilerPass<MpsModule<'a>> for MergeDots {
    fn name(&self) -> &'static str {
        "merge_dots"
    }

    fn apply(&mut self, m: &mut MpsModule<'a>) -> Result<(), String> {
        let options = m.options;
        let donated: Vec<bool> = (0..options.packable.len())
            .map(|i| {
                options.donate.contains(&i) || options.donate_into.iter().any(|&(j, _)| j == i)
            })
            .collect();
        (m.graph, m.packed, m.interleaved) = merge_dots(&m.graph, &options.packable, &donated);
        Ok(())
    }
}
