use super::pass::{CompilerPass, Module, Pipeline};
use super::passes::dot_strength::DotStrength;
use super::passes::fold_dot_reshapes::FoldDotReshapes;
use super::passes::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::passes::simplify::Simplify;
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::{Graph, Node, Primitive, Var, intern};
use crate::graph_compiler::{Plan, PlanOptions};

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

fn columns(graph: &Graph, dot: &Node) -> Option<Vec<(usize, Var)>> {
    if !matches!(dot.primitive, Primitive::DotGeneral { .. })
        || graph.outputs().contains(&dot.output)
    {
        return None;
    }
    let shape = &graph.type_of(dot.output).shape;
    let last = shape.len() - 1;
    let mut parts: Vec<(usize, usize, Var)> = Vec::new();
    for node in graph
        .nodes()
        .iter()
        .filter(|n| n.inputs.contains(&dot.output))
    {
        let Primitive::Slice {
            start_indices,
            limit_indices,
            ..
        } = &node.primitive
        else {
            return None;
        };
        let whole = (0..last).all(|d| start_indices[d] == 0 && limit_indices[d] == shape[d]);
        if !whole {
            return None;
        }
        parts.push((start_indices[last], limit_indices[last], node.output));
    }
    parts.sort_unstable();
    let covers = parts.len() > 1
        && parts.first().is_some_and(|p| p.0 == 0)
        && parts.windows(2).all(|w| w[0].1 == w[1].0)
        && parts.last().is_some_and(|p| p.1 == shape[last]);
    covers.then(|| parts.into_iter().map(|(a, b, v)| (b - a, v)).collect())
}

fn split_dots(graph: &Graph) -> Graph {
    let mut out = Graph::new();
    let mut map: Vec<Var> = vec![0; graph.types.len()];
    for &v in graph.inputs() {
        map[v] = out.input(graph.type_of(v).clone());
    }
    let mut parted = vec![false; graph.types.len()];
    for node in graph.nodes() {
        out.set_origin(node);
        if parted[node.output] {
            continue;
        }
        if let Some(parts) = columns(graph, node) {
            let mut body = Graph::new();
            let operands: Vec<Var> = node
                .inputs
                .iter()
                .map(|&v| body.input(graph.type_of(v).clone()))
                .collect();
            let dot = body
                .apply(node.primitive.clone(), &operands)
                .expect("the dot's operands");
            let shape = body.type_of(dot).shape.clone();
            let (mut start, mut outputs) = (0, Vec::new());
            for &(width, _) in &parts {
                let (mut starts, mut limits) = (vec![0; shape.len()], shape.clone());
                (starts[shape.len() - 1], limits[shape.len() - 1]) = (start, start + width);
                let slice = Primitive::Slice {
                    start_indices: starts,
                    limit_indices: limits,
                    view: false,
                };
                outputs.push(body.apply(slice, &[dot]).expect("within the dot"));
                start += width;
            }
            body.set_outputs(&outputs).expect("the body's values");
            let label = intern(format!("{}x dot_general", parts.len()));
            let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
            let fused = out
                .apply(
                    Primitive::Fusion {
                        name: "dot_general".into(),
                        label,
                        body,
                    },
                    &inputs,
                )
                .expect("a fusion of the dot");
            map[parts[0].1] = fused;
            for (index, &(_, v)) in parts.iter().enumerate().skip(1) {
                let ty = graph.type_of(v).clone();
                map[v] = out
                    .apply(Primitive::FusionOutput { index, ty }, &[fused])
                    .expect("the fusion's output");
            }
            parts.iter().for_each(|&(_, v)| parted[v] = true);
            continue;
        }
        let inputs: Vec<Var> = node.inputs.iter().map(|&v| map[v]).collect();
        map[node.output] = out
            .apply(node.primitive.clone(), &inputs)
            .expect("the node's own operands");
        out.set_label(map[node.output], node.label);
    }
    let outputs: Vec<Var> = graph.outputs().iter().map(|&v| map[v]).collect();
    out.set_outputs(&outputs)
        .expect("outputs are values of the graph");
    out
}

struct SplitDots;

impl CompilerPass<TpuModule<'_>> for SplitDots {
    fn name(&self) -> &'static str {
        "split_dots"
    }

    fn apply(&mut self, m: &mut TpuModule<'_>) -> Result<(), String> {
        m.graph = split_dots(&m.graph);
        Ok(())
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
    pipeline
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .add(DotStrength { matrices: false });
    if options.config.merge_dots {
        pipeline
            .add(MergeDots {
                target: Target {
                    reads: |_, _| true,
                    views: false,
                    lanes: 128,
                    columns: 1,
                },
            })
            .add(SplitDots);
    }
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
