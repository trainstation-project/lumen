mod diamonds;
mod flash_attention_backward;
mod fusion;
mod place_concats;
#[cfg(test)]
mod tests;

use diamonds::Diamonds;
use flash_attention_backward::FlashAttentionBackward;
use fusion::{DotRows, FuseDots, HorizontalFusion, LoopFusion};
use place_concats::PlaceConcats;

use super::common::attention;
use super::common::clamp_indices::ClampIndices;
use super::common::dot_strength::DotStrength;
use super::common::flash_attention::{Attentions, FlashAttention};
use super::common::fold_dot_reshapes::FoldDotReshapes;
use super::common::merge_dots::{Blocks, MergeDots, Packs, Target};
use super::common::merge_dots_backward::MergeDotsBackward;
use super::common::simplify::Simplify;
use super::pass::{Module, Pipeline};
use super::plan::DeviceLayout;
use super::{Options, positions};
use crate::graph::{Graph, Var};
use crate::graph_compiler::{Plan, PlanOptions};

const LANES: usize = 128;

const LAYOUT: DeviceLayout = DeviceLayout {
    bytes: |t| {
        crate::device::tpu::layout::bytes(t.dtype, &t.shape)
            .unwrap_or(t.numel() * t.dtype.size_of())
    },
    same: crate::device::tpu::layout::same,
    align: |t| crate::device::tpu::layout::alignment(t.dtype, &t.shape),
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

impl Attentions for TpuModule<'_> {
    fn epilogues(&self) -> attention::Epilogues {
        attention::Epilogues {
            leaves: |graph, v| graph.type_of(v).shape.is_empty(),
            reshaped: true,
        }
    }

    fn scalars(&self) -> Vec<Var> {
        Vec::new()
    }

    fn kernel(&mut self) -> String {
        "flash_attention".into()
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
    if options.config.safe_kernels {
        pipeline.add(ClampIndices);
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
        pipeline.add(Diamonds).add(DotRows);
    }
    pipeline.add(FuseDots).add(PlaceConcats).add(LoopFusion);
    if options.config.horizontal_fusion {
        pipeline.add(HorizontalFusion);
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
