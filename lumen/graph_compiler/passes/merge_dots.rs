use std::collections::HashMap;

use crate::codegen::mps::ROW_COLUMNS;
use crate::graph::primitive::free_dims;
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::graph_compiler::pass::{CompilerPass, Module};
use crate::graph_compiler::{combined, elementwise};

pub(crate) struct Target {
    pub(crate) reads: fn(&Graph, &Node) -> bool,
    pub(crate) views: bool,
    pub(crate) lanes: usize,
}

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

pub(crate) type Packs = Vec<(Vec<usize>, usize)>;

pub(crate) fn merge_dots(
    graph: &Graph,
    packable: &[bool],
    donated: &[bool],
    target: &Target,
) -> (Graph, Packs, Vec<Vec<usize>>) {
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
        (target.reads)(graph, node) || matches!(node.primitive, Primitive::DotGeneral { .. })
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
            let tiled = other
                .shape
                .iter()
                .rev()
                .take(2)
                .all(|n| n.is_multiple_of(target.lanes));
            let whole = side == 0 && !strided && (chain.is_empty() || alone[w]) && tiled;
            if !target.views && !whole {
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
    let mut groups: Vec<Group> = groups
        .into_iter()
        .flat_map(
            |g| match interleaves(graph, &g, &position, &alone, &reads, donated) {
                Some((order, rest, ..)) if target.views && !rest.is_empty() => {
                    vec![g.subset(&order), g.subset(&rest)]
                }
                _ => vec![g],
            },
        )
        .collect();
    let mut merged = vec![false; nodes.len()];
    let mut first_of: Vec<Option<usize>> = vec![None; nodes.len()];
    let mut packed = vec![false; graph.types.len()];
    let mut reuses: Vec<Option<usize>> = vec![None; groups.len()];
    let mut layouts: Vec<Option<Layout>> = vec![None; groups.len()];
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
            layouts[k] = layouts[e].clone();
            continue;
        }
        let g = &mut groups[k];
        g.nodes.retain(|&i| !packed[weight(side, i)]);
        if g.nodes.len() > 1 {
            for &i in &g.nodes {
                merged[i] = true;
                packed[weight(side, i)] = true;
            }
            if target.views
                && let Some((order, rest, layout, rewrites)) =
                    interleaves(graph, g, &position, &alone, &reads, donated)
                && rest.is_empty()
            {
                g.nodes = order.iter().map(|&j| g.nodes[j]).collect();
                g.weights = order.iter().map(|&j| g.weights[j]).collect();
                layouts[k] = Some(layout);
                for &c in &rewrites.cotangents {
                    cotangents[c] = Some(k);
                }
                for &(s, j, axis) in &rewrites.slices {
                    pairs_of[s] = Some((k, j, axis));
                }
            }
            first_of[g.nodes[0]] = Some(k);
        }
    }

    let mut packs = Vec::new();
    let mut interleaved = Vec::new();
    let mut tupled: HashMap<(Var, usize, usize), Var> = HashMap::new();
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
                    interleaved.push(
                        layouts[k]
                            .as_ref()
                            .map_or(vec![1; g.nodes.len()], |l| l.regions.clone()),
                    );
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
            if layouts[k].is_none() && !g.chain.is_empty() {
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
            if let Some(layout) = &layouts[k] {
                for (j, &m) in g.nodes.iter().enumerate() {
                    out.set_origin(&nodes[m]);
                    let member = &graph.type_of(nodes[m].output).shape;
                    map[nodes[m].output] =
                        member_of(&mut out, &mut tupled, result, dimension, layout, j, member);
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
            Some(k),
            Primitive::Concatenate {
                dimension: axis, ..
            },
        ) = (cotangents[i], &node.primitive)
        {
            let layout = layouts[k].as_ref().expect("an interleaved group");
            let parts: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = arrange(&mut out, &parts, *axis, layout);
            continue;
        }
        if let Some((k, j, axis)) = pairs_of[i] {
            let layout = layouts[k].as_ref().expect("an interleaved group");
            let shape = &graph.type_of(node.output).shape;
            map[node.output] = member_of(
                &mut out,
                &mut tupled,
                map[node.inputs[0]],
                axis,
                layout,
                j,
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

impl Group {
    fn subset(&self, members: &[usize]) -> Group {
        Group {
            shared: self.shared,
            side: self.side,
            primitive: self.primitive.clone(),
            other: self.other.clone(),
            dimension: self.dimension,
            chain: self.chain.clone(),
            nodes: members.iter().map(|&k| self.nodes[k]).collect(),
            weights: members.iter().map(|&k| self.weights[k]).collect(),
        }
    }
}

#[derive(Clone, Debug)]
struct Layout {
    regions: Vec<usize>,
    sizes: Vec<usize>,
}

impl Layout {
    fn region_of(&self, j: usize) -> (usize, usize, usize) {
        let (mut member, mut offset) = (0, 0);
        for &k in &self.regions {
            if j < member + k {
                return (offset, k, j - member);
            }
            offset += k * self.sizes[member];
            member += k;
        }
        unreachable!("a member of the layout")
    }
}

fn member_of(
    out: &mut Graph,
    cache: &mut HashMap<(Var, usize, usize), Var>,
    value: Var,
    axis: usize,
    layout: &Layout,
    j: usize,
    shape: &[usize],
) -> Var {
    let (offset, k, within) = layout.region_of(j);
    let n = layout.sizes[j];
    let full = out.type_of(value).shape.clone();
    let region = match layout.regions.len() {
        1 => value,
        _ => *cache.entry((value, axis, offset)).or_insert_with(|| {
            let (mut starts, mut limits) = (vec![0; full.len()], full.clone());
            (starts[axis], limits[axis]) = (offset, offset + k * n);
            let slice = Primitive::Slice {
                start_indices: starts,
                limit_indices: limits,
                view: false,
            };
            out.apply(slice, &[value]).expect("within the value")
        }),
    };
    if k == 1 {
        return region;
    }
    let mut tuples = out.type_of(region).shape.clone();
    tuples.splice(axis..=axis, [n, k]);
    let reshaped = *cache.entry((region, axis, usize::MAX)).or_insert_with(|| {
        let reshape = Primitive::Reshape {
            new_sizes: tuples.clone(),
        };
        out.apply(reshape, &[region]).expect("the same elements")
    });
    let (mut starts, mut limits) = (vec![0; tuples.len()], tuples);
    (starts[axis + 1], limits[axis + 1]) = (within, within + 1);
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

fn arrange(out: &mut Graph, parts: &[Var], axis: usize, layout: &Layout) -> Var {
    let mut regions = Vec::new();
    let mut member = 0;
    for &k in &layout.regions {
        let group = &parts[member..member + k];
        member += k;
        if k == 1 {
            regions.push(group[0]);
            continue;
        }
        let tuples: Vec<Var> = group
            .iter()
            .map(|&v| {
                let mut one = out.type_of(v).shape.clone();
                one.insert(axis + 1, 1);
                let reshape = Primitive::Reshape { new_sizes: one };
                out.apply(reshape, &[v]).expect("the same elements")
            })
            .collect();
        let concatenated = Primitive::Concatenate {
            dimension: axis + 1,
            in_place: false,
        };
        let tuples = out
            .apply(concatenated, &tuples)
            .expect("parts of one shape");
        let mut shape = out.type_of(group[0]).shape.clone();
        shape[axis] *= k;
        let back = Primitive::Reshape { new_sizes: shape };
        regions.push(out.apply(back, &[tuples]).expect("the same elements"));
    }
    match regions.as_slice() {
        [one] => *one,
        _ => {
            let concatenated = Primitive::Concatenate {
                dimension: axis,
                in_place: false,
            };
            out.apply(concatenated, &regions)
                .expect("regions of one shape but the axis")
        }
    }
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
) -> Option<(Vec<usize>, Vec<usize>, Layout, Interleaving)> {
    let nodes = graph.nodes();
    let operand = |i: usize| nodes[i].inputs[1 - g.side];
    let size = |w: Var| graph.type_of(w).shape[g.dimension];
    let fits = g.side == 0
        && g.weights.iter().all(|&w| {
            let i = position[w].expect("a packable input");
            alone[w] && donated.get(i) != Some(&true)
        })
        && g.nodes.iter().all(|&i| reads[operand(i)] == 1);
    if !fits {
        return None;
    }
    let results: Vec<Var> = g.nodes.iter().map(|&i| nodes[i].output).collect();
    let sets = combined(graph, &results);
    let mut by_set: Vec<(usize, Vec<usize>)> = Vec::new();
    for (j, &set) in sets.iter().enumerate() {
        match by_set.iter_mut().find(|(s, _)| *s == set) {
            Some((_, members)) => members.push(j),
            None => by_set.push((set, vec![j])),
        }
    }
    let width = |m: &Vec<usize>| m.len() * size(g.weights[m[0]]);
    let tiled = |m: &Vec<usize>| {
        ROW_COLUMNS.is_multiple_of(m.len())
            && m.iter()
                .all(|&j| size(g.weights[j]) == size(g.weights[m[0]]))
            && width(m).is_multiple_of(ROW_COLUMNS)
    };
    let mut groups: Vec<&Vec<usize>> = by_set
        .iter()
        .map(|(_, m)| m)
        .filter(|m| m.len() > 1 && tiled(m))
        .collect();
    if groups.is_empty() {
        return None;
    }
    groups.sort_by_key(|m| std::cmp::Reverse(m.len()));
    let singles: Vec<&Vec<usize>> = by_set
        .iter()
        .map(|(_, m)| m)
        .filter(|m| m.len() == 1 && tiled(m))
        .collect();
    let order: Vec<usize> = groups
        .iter()
        .chain(&singles)
        .flat_map(|m| m.iter().copied())
        .collect();
    let rest: Vec<usize> = (0..g.nodes.len()).filter(|j| !order.contains(j)).collect();
    let regions: Vec<usize> = groups.iter().chain(&singles).map(|m| m.len()).collect();
    let weights: Vec<Var> = order.iter().map(|&j| g.weights[j]).collect();
    let layout = Layout {
        regions,
        sizes: weights.iter().map(|&w| size(w)).collect(),
    };
    let operands: Vec<Var> = order.iter().map(|&j| operand(g.nodes[j])).collect();
    let rewrites = backward(graph, g.dimension, &weights, &operands, &layout)?;
    Some((order, rest, layout, rewrites))
}

fn backward(
    graph: &Graph,
    dimension: usize,
    weights: &[Var],
    operands: &[Var],
    layout: &Layout,
) -> Option<Interleaving> {
    let nodes = graph.nodes();
    let k = weights.len();
    let outputs = graph.outputs();
    let readers = |v: Var| {
        nodes
            .iter()
            .enumerate()
            .filter(move |(_, n)| n.inputs.contains(&v))
    };
    let produced = |v: Var| nodes.iter().position(|n| n.output == v);
    let concatenate_of_parts = |n: &Node, axis: usize| {
        matches!(n.primitive, Primitive::Concatenate { dimension, .. } if dimension == axis)
            && n.inputs.len() == k
            && n.inputs
                .iter()
                .zip(&layout.sizes)
                .all(|(&v, &size)| graph.type_of(v).shape[axis] == size)
    };
    let mut found = Interleaving::default();
    let mut contractions = Vec::new();
    for b in nodes.iter().filter(|n| {
        matches!(n.primitive, Primitive::Concatenate { dimension: d, .. } if d == dimension)
            && (n.inputs == weights || n.inputs == operands)
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
            let c = mine.iter().position(|&d| d == dimension)?;
            let cotangent = &nodes[produced(reader.inputs[1 - side])?];
            if !concatenate_of_parts(cotangent, theirs[c]) {
                return None;
            }
            let at = produced(cotangent.output)?;
            if !found.cotangents.contains(&at) {
                found.cotangents.push(at);
            }
            contractions.push(r);
        }
    }
    let offsets: Vec<usize> = layout
        .sizes
        .iter()
        .scan(0, |start, &n| {
            let at = *start;
            *start += n;
            Some(at)
        })
        .collect();
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
                let j = offsets
                    .iter()
                    .position(|&o| o == start_indices[result_axis])?;
                let part = limit_indices[result_axis] == offsets[j] + layout.sizes[j]
                    && (0..result.shape.len()).all(|d| d == result_axis || whole(d));
                if !part || members[j] {
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

pub(crate) trait Blocks {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>);
}

pub(crate) struct MergeDots {
    pub(crate) target: Target,
}

impl<T: Module + Blocks> CompilerPass<T> for MergeDots {
    fn name(&self) -> &'static str {
        "merge_dots"
    }

    fn apply(&mut self, m: &mut T) -> Result<(), String> {
        let options = m.options();
        let donated: Vec<bool> = (0..options.packable.len())
            .map(|i| {
                options.donate.contains(&i) || options.donate_into.iter().any(|&(j, _)| j == i)
            })
            .collect();
        let (graph, packed, interleaved) =
            merge_dots(m.graph(), &options.packable, &donated, &self.target);
        *m.graph_mut() = graph;
        let (p, i) = m.blocks();
        (*p, *i) = (packed, interleaved);
        Ok(())
    }
}
