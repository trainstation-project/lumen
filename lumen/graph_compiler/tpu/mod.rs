use std::collections::HashMap;

use super::attention;
use super::pass::{CompilerPass, Module, Pipeline};
use super::passes::dot_strength::DotStrength;
use super::passes::fold_dot_reshapes::FoldDotReshapes;
use super::passes::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::passes::merge_dots_backward::MergeDotsBackward;
use super::passes::simplify::Simplify;
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::{FUSION_SEPARATOR, Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::elementwise;
use crate::graph_compiler::{Plan, PlanOptions};

const LANES: usize = 128;

const MAX_TOGETHER: usize = 4;

const LAYOUT: DeviceLayout = DeviceLayout {
    bytes: |t| {
        crate::device::tpu::layout::bytes(t.dtype, &t.shape)
            .unwrap_or(t.numel() * t.dtype.size_of())
    },
    same: crate::device::tpu::layout::same,
};

struct TpuModule<'a> {
    graph: Graph,
    options: &'a Options,
    packed: Packs,
    interleaved: Vec<Vec<usize>>,
}

impl Module for TpuModule<'_> {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn options(&self) -> &Options {
        self.options
    }
}

impl Blocks for TpuModule<'_> {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

fn split(graph: &Graph, dot: &Node) -> Option<(usize, Vec<(usize, Var)>)> {
    if graph.outputs().contains(&dot.output) {
        return None;
    }
    let shape = &graph.type_of(dot.output).shape;
    let readers: Vec<&Node> = graph
        .nodes()
        .iter()
        .filter(|n| n.inputs.contains(&dot.output))
        .collect();
    let along = |axis: usize| {
        let mut parts: Vec<(usize, usize, Var)> = Vec::new();
        for node in &readers {
            let Primitive::Slice {
                start_indices,
                limit_indices,
                ..
            } = &node.primitive
            else {
                return None;
            };
            let whole = (0..shape.len())
                .filter(|&d| d != axis)
                .all(|d| start_indices[d] == 0 && limit_indices[d] == shape[d]);
            if !whole {
                return None;
            }
            parts.push((start_indices[axis], limit_indices[axis], node.output));
        }
        parts.sort_unstable();
        let covers = parts.len() > 1
            && parts.first().is_some_and(|p| p.0 == 0)
            && parts.windows(2).all(|w| w[0].1 == w[1].0)
            && parts.last().is_some_and(|p| p.1 == shape[axis])
            && parts
                .iter()
                .all(|p| p.0.is_multiple_of(LANES) && p.1.is_multiple_of(LANES));
        covers.then(|| {
            (
                axis,
                parts.into_iter().map(|(a, b, v)| (b - a, v)).collect(),
            )
        })
    };
    along(shape.len() - 1).or_else(|| along(shape.len() - 2))
}

fn stacked(
    graph: &Graph,
    producer: &[Option<usize>],
    users: &[Vec<usize>],
    dot: &Node,
) -> Option<usize> {
    let lhs = dot.inputs[0];
    let k = producer[lhs]?;
    let Primitive::Concatenate { dimension, .. } = graph.nodes()[k].primitive else {
        return None;
    };
    let rank = graph.type_of(lhs).shape.len();
    let dots = users[lhs].iter().all(|&u| {
        let reader = &graph.nodes()[u];
        matches!(reader.primitive, Primitive::DotGeneral { .. })
            && reader.inputs[0] == lhs
            && reader.inputs[1] != lhs
    });
    let fits = dots
        && !placeable(graph, producer, users, &graph.nodes()[k])
        && !graph.outputs().contains(&lhs)
        && dimension + 2 >= rank
        && graph.nodes()[k]
            .inputs
            .iter()
            .all(|&v| graph.type_of(v).shape[dimension].is_multiple_of(LANES));
    fits.then_some(k)
}

struct Fused {
    parts: Option<(usize, Vec<(usize, Var)>)>,
    stack: Option<usize>,
    inside: Vec<usize>,
    extras: Vec<Var>,
    outputs: Vec<Var>,
}

fn fused(graph: &Graph, epilogues: bool) -> Vec<Option<Fused>> {
    let nodes = graph.nodes();
    let n = graph.types.len();
    let mut producer: Vec<Option<usize>> = vec![None; n];
    let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        producer[node.output] = Some(i);
        node.inputs.iter().for_each(|&v| users[v].push(i));
    }
    let mut claimed = vec![false; nodes.len()];
    let mut found: Vec<Option<Fused>> = (0..nodes.len()).map(|_| None).collect();
    for (d, dot) in nodes.iter().enumerate() {
        if !matches!(dot.primitive, Primitive::DotGeneral { .. }) {
            continue;
        }
        let before = |v: Var| producer[v].is_none_or(|p| p < d);
        let parts = split(graph, dot);
        let stack = stacked(graph, &producer, &users, dot);
        let mut member: HashMap<Var, usize> = match &parts {
            Some((_, parts)) => parts
                .iter()
                .enumerate()
                .map(|(k, &(_, v))| (v, k))
                .collect(),
            None => HashMap::from([(dot.output, 0)]),
        };
        let shape_of = |p: usize| match &parts {
            Some((_, parts)) => &graph.type_of(parts[p].1).shape,
            None => &graph.type_of(dot.output).shape,
        };
        let together = parts.as_ref().is_some_and(|(axis, parts)| {
            *axis + 1 == graph.type_of(dot.output).shape.len()
                && parts.len() <= MAX_TOGETHER
                && parts.iter().all(|&(w, _)| w == parts[0].0)
        });
        let (mut inside, mut extras, mut absorbed) = (Vec::new(), Vec::new(), Vec::new());
        let mut read_by: HashMap<Var, usize> = HashMap::new();
        for (u, un) in nodes.iter().enumerate().skip(d + 1) {
            if !epilogues || claimed[u] || !elementwise(&un.primitive) {
                continue;
            }
            let mut of: Vec<usize> = un
                .inputs
                .iter()
                .filter_map(|v| member.get(v))
                .copied()
                .collect();
            of.sort_unstable();
            of.dedup();
            let Some(&p) = of.first() else {
                continue;
            };
            if of.len() > 1 && !together {
                continue;
            }
            if graph.type_of(un.output).shape != *shape_of(p) {
                continue;
            }
            let (mut constants, mut operands) = (Vec::new(), Vec::new());
            let fits = un
                .inputs
                .iter()
                .filter(|v| !member.contains_key(v))
                .all(|&v| {
                    let ty = graph.type_of(v);
                    let made = producer[v].map(|k| &nodes[k]);
                    match made.map(|m| &m.primitive) {
                        Some(Primitive::Full { .. }) => {
                            constants.push(producer[v].expect("made"));
                            true
                        }
                        Some(Primitive::BroadcastInDim {
                            broadcast_dimensions,
                            ..
                        }) if broadcast_dimensions.is_empty() => {
                            let x = made.expect("made").inputs[0];
                            constants.push(producer[v].expect("made"));
                            match producer[x].map(|k| &nodes[k].primitive) {
                                Some(Primitive::Full { .. }) => {
                                    constants.push(producer[x].expect("made"));
                                    true
                                }
                                _ if before(x) => {
                                    operands.push(x);
                                    true
                                }
                                _ => false,
                            }
                        }
                        _ => before(v) && (ty.shape.is_empty() || ty.shape == *shape_of(p)),
                    }
                });
            for v in un.inputs.iter().filter(|v| !member.contains_key(v)) {
                let constant = producer[*v].is_some_and(|k| constants.contains(&k));
                if !constant && !operands.contains(v) {
                    operands.push(*v);
                }
            }
            let shared = operands.iter().any(|v| {
                !together
                    && !graph.type_of(*v).shape.is_empty()
                    && read_by.get(v).is_some_and(|&q| q != p)
            });
            if !fits || shared {
                continue;
            }
            for &v in &operands {
                read_by.entry(v).or_insert(p);
            }
            claimed[u] = true;
            member.insert(un.output, p);
            absorbed.push(u);
            for k in constants.into_iter().chain([u]) {
                if !inside.contains(&k) {
                    inside.push(k);
                }
            }
            for v in operands {
                if !extras.contains(&v) {
                    extras.push(v);
                }
            }
        }
        if parts.is_none() && stack.is_none() && absorbed.is_empty() {
            continue;
        }
        inside.sort_unstable();
        let read_outside =
            |v: Var| graph.outputs().contains(&v) || users[v].iter().any(|u| !absorbed.contains(u));
        let mut outputs: Vec<Var> = member
            .keys()
            .copied()
            .filter(|&v| read_outside(v))
            .collect();
        outputs.sort_unstable();
        if outputs.is_empty() {
            continue;
        }
        found[d] = Some(Fused {
            parts,
            stack,
            inside,
            extras,
            outputs,
        });
    }
    found
}

fn fuse_dots(graph: &Graph, epilogues: bool) -> Graph {
    let nodes = graph.nodes();
    let found = fused(graph, epilogues);
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut skipped = vec![false; nodes.len()];
    for f in found.iter().flatten() {
        if let Some(k) = f.stack {
            skipped[k] = true;
        }
        f.inside
            .iter()
            .filter(|&&k| {
                !matches!(
                    nodes[k].primitive,
                    Primitive::Full { .. } | Primitive::BroadcastInDim { .. }
                )
            })
            .for_each(|&k| skipped[k] = true);
    }
    let mut parted = vec![false; graph.types.len()];
    for (d, node) in nodes.iter().enumerate() {
        out.set_origin(node);
        if skipped[d] || parted[node.output] {
            continue;
        }
        let Some(f) = &found[d] else {
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            map[node.output] = out
                .apply(node.primitive.clone(), &inputs)
                .expect("the node's own operands");
            out.set_label(map[node.output], node.label);
            continue;
        };
        let mut body = Graph::new();
        let mut within: HashMap<Var, Var> = HashMap::new();
        let reads: Vec<Var> = match f.stack {
            Some(k) => nodes[k]
                .inputs
                .iter()
                .chain(&node.inputs[1..])
                .copied()
                .collect(),
            None => node.inputs.clone(),
        };
        for &v in reads.iter().chain(&f.extras) {
            within.insert(v, body.input(graph.type_of(v).clone()));
        }
        if let Some(k) = f.stack {
            let parts: Vec<Var> = nodes[k].inputs.iter().map(|v| within[v]).collect();
            let v = body
                .apply(nodes[k].primitive.clone(), &parts)
                .expect("the stacked operands");
            within.insert(nodes[k].output, v);
        }
        let operands: Vec<Var> = node.inputs.iter().map(|v| within[v]).collect();
        let dot = body
            .apply(node.primitive.clone(), &operands)
            .expect("the dot's operands");
        within.insert(node.output, dot);
        if let Some((axis, parts)) = &f.parts {
            let shape = body.type_of(dot).shape.clone();
            let mut start = 0;
            for &(width, v) in parts {
                let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                (starts[*axis], limits[*axis]) = (start, start + width);
                let slice = Primitive::Slice {
                    start_indices: starts,
                    limit_indices: limits,
                    view: false,
                };
                within.insert(v, body.apply(slice, &[dot]).expect("within the dot"));
                parted[v] = true;
                start += width;
            }
        }
        let mut names = Vec::new();
        for &k in &f.inside {
            let inputs: Vec<Var> = nodes[k].inputs.iter().map(|v| within[v]).collect();
            let v = body
                .apply(nodes[k].primitive.clone(), &inputs)
                .expect("the epilogue's operands");
            within.insert(nodes[k].output, v);
            if skipped[k] {
                names.push(graph.label(&nodes[k]));
            }
        }
        let outputs: Vec<Var> = f.outputs.iter().map(|v| within[v]).collect();
        body.set_outputs(&outputs).expect("the body's values");
        let members = match (&f.parts, f.stack) {
            (Some((_, parts)), _) => parts.len(),
            (None, Some(k)) => nodes[k].inputs.len(),
            (None, None) => 1,
        };
        let mut label = match members > 1 && node.label.is_none() {
            true => format!("{members}x {}", graph.label(node)),
            false => graph.label(node).to_string(),
        };
        for name in names {
            label += &format!("{FUSION_SEPARATOR}{name}");
        }
        let inputs: Vec<Var> = reads.iter().chain(&f.extras).map(|&v| map[v]).collect();
        let fused = out
            .apply(
                Primitive::Fusion {
                    name: "dot_general".into(),
                    label: intern(label),
                    body,
                },
                &inputs,
            )
            .expect("a fusion of the dot");
        map[f.outputs[0]] = fused;
        for (index, &v) in f.outputs.iter().enumerate().skip(1) {
            let ty = graph.type_of(v).clone();
            map[v] = out
                .apply(Primitive::FusionOutput { index, ty }, &[fused])
                .expect("the fusion's output");
        }
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    out
}

fn placeable(
    graph: &Graph,
    producer: &[Option<usize>],
    users: &[Vec<usize>],
    concat: &Node,
) -> bool {
    let nodes = graph.nodes();
    let Primitive::Concatenate { dimension, .. } = concat.primitive else {
        return false;
    };
    let writes = |p: &Primitive| match p {
        Primitive::DotGeneral { .. } | Primitive::Full { .. } => true,
        Primitive::Fusion { name, .. } => name == "dot_general",
        Primitive::BroadcastInDim {
            broadcast_dimensions,
            ..
        } => broadcast_dimensions.is_empty(),
        p => elementwise(p),
    };
    dimension + 1 == graph.type_of(concat.output).shape.len()
        && concat.inputs.iter().all(|&v| {
            let others = users[v]
                .iter()
                .filter(|&&u| !matches!(nodes[u].primitive, Primitive::FusionOutput { .. }));
            let alone = others.count() == 1 && !graph.outputs().contains(&v);
            let once = concat.inputs.iter().filter(|&&u| u == v).count() == 1;
            let made = producer[v].map(|p| match nodes[p].primitive {
                Primitive::FusionOutput { .. } => producer[nodes[p].inputs[0]],
                _ => Some(p),
            });
            alone
                && once
                && graph.type_of(v).shape[dimension].is_multiple_of(LANES)
                && made.flatten().is_some_and(|p| writes(&nodes[p].primitive))
        })
}

struct PlaceConcats;

impl CompilerPass<TpuModule<'_>> for PlaceConcats {
    fn name(&self) -> &'static str {
        "place_concats"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        let nodes = m.graph.nodes();
        let mut producer: Vec<Option<usize>> = vec![None; m.graph.types.len()];
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); m.graph.types.len()];
        for (i, node) in nodes.iter().enumerate() {
            producer[node.output] = Some(i);
            node.inputs.iter().for_each(|&v| users[v].push(i));
        }
        let placed: Vec<usize> = (0..nodes.len())
            .filter(|&c| placeable(&m.graph, &producer, &users, &nodes[c]))
            .collect();
        for c in placed {
            if let Primitive::Concatenate { in_place, .. } = &mut m.graph.nodes[c].primitive {
                *in_place = true;
            }
        }
        Ok(())
    }
}

struct FuseDots;

impl CompilerPass<TpuModule<'_>> for FuseDots {
    fn name(&self) -> &'static str {
        "fuse_dots"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = fuse_dots(&m.graph, m.options.config.contraction_epilogues);
        Ok(())
    }
}

fn attentions(graph: &Graph) -> Result<(), String> {
    for node in graph.nodes() {
        if let Primitive::Fusion { name, body, .. } = &node.primitive {
            match name.as_str() {
                "flash_attention" => crate::ops::attention::tpu::forward(body).map(drop)?,
                "flash_attention_backward" => {
                    crate::ops::attention::tpu::backward(body).map(drop)?
                }
                _ => {}
            }
        }
    }
    Ok(())
}

struct FlashAttention;

impl CompilerPass<TpuModule<'_>> for FlashAttention {
    fn name(&self) -> &'static str {
        "flash_attention"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = attention::fuse(&m.graph, false, &[], |_| "flash_attention".into());
        attentions(&m.graph)
    }
}

struct FlashAttentionBackward;

impl CompilerPass<TpuModule<'_>> for FlashAttentionBackward {
    fn name(&self) -> &'static str {
        "flash_attention_backward"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph =
            attention::fuse_backward(&m.graph, |_| false, |_| "flash_attention_backward".into());
        attentions(&m.graph)
    }
}

pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = TpuModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        interleaved: Vec::new(),
    };
    let mut pipeline = Pipeline::new("tpu");
    if options.config.merge_dots {
        pipeline.add(MergeDotsBackward);
    }
    if options.config.flash_attention {
        pipeline.add(FlashAttention).add(FlashAttentionBackward);
    }
    pipeline
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .add(DotStrength { matrices: false });
    if options.config.merge_dots {
        pipeline.add(MergeDots {
            target: Target {
                reads: |_, _| true,
                views: false,
                lanes: LANES,
                columns: 1,
            },
        });
    }
    pipeline.add(FuseDots).add(PlaceConcats);
    pipeline.run(&mut module)?;
    let TpuModule {
        graph,
        packed,
        interleaved,
        ..
    } = module;
    let parameters = options.parameters.clone().map(|mut p| {
        p.resize(graph.inputs().len(), true);
        p
    });
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters,
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
        blocks: packed
            .iter()
            .enumerate()
            .map(|(k, (members, _))| (graph.inputs().len() - packed.len() + k, members.clone()))
            .collect(),
        layout: Some(LAYOUT),
    };
    let mut plan = Plan::compile_with(&graph, &plan);
    plan.packed = packed;
    plan.interleaved = interleaved;
    Ok(plan)
}
