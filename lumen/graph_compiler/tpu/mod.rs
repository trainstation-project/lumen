mod diamonds;
mod flash_attention;
mod flash_attention_backward;
mod fusion;
mod place_concats;

use diamonds::Diamonds;
use flash_attention::FlashAttention;
use flash_attention_backward::FlashAttentionBackward;
use fusion::{FuseDots, LoopFusion};
use place_concats::PlaceConcats;

use super::pass::{Module, Pipeline};
use super::passes::dot_strength::DotStrength;
use super::passes::fold_dot_reshapes::FoldDotReshapes;
use super::passes::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::passes::merge_dots_backward::MergeDotsBackward;
use super::passes::simplify::Simplify;
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::Graph;
use crate::graph_compiler::{Plan, PlanOptions};

const LANES: usize = 128;

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
    if options.config.normalization_diamonds {
        pipeline.add(Diamonds);
    }
    pipeline.add(FuseDots).add(PlaceConcats).add(LoopFusion);
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
