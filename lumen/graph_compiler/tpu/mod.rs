//! TPU graph compiler: the pass pipeline that rewrites a traced graph into TPU kernels and plans its buffers.
mod diamonds;
mod flash_attention_backward;
mod fusion;
mod place_concats;
#[cfg(test)]
mod tests;

use super::common::updates::Updates;
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
use super::{CompilerConfig, Options, positions};
use crate::graph::{Graph, Var};
use crate::graph_compiler::{Plan, PlanOptions};

/// Vector lane width of a TPU register; concatenated and merged dot widths are aligned to it.
const LANES: usize = 128;

/// Buffer layout of TPU tensors, which are stored TPU-tiled in the shared region.
const LAYOUT: DeviceLayout = DeviceLayout {
    bytes: |t| {
        // Shapes the tiled layout can't describe fall back to their dense size.
        crate::device::tpu::layout::bytes(t.dtype, &t.shape)
            .unwrap_or(t.numel() * t.dtype.size_of())
    },
    same: crate::device::tpu::layout::same,
    align: |t| crate::device::tpu::layout::alignment(t.dtype, &t.shape),
};

/// Graph being compiled plus the state the TPU passes carry between them.
struct TpuModule<'a> {
    graph: Graph,
    options: &'a Options,
    /// Packed weight blocks built by dot merging, appended as the graph's last inputs.
    packed: Packs,
    /// Per-pack region layout of interleaved members, parallel to `packed`.
    interleaved: Vec<Vec<usize>>,
}

/// Exposes the graph and the TPU compiler config to the shared pass framework.
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

    fn config(&self) -> &dyn CompilerConfig {
        &self.options.config.tpu
    }
}

/// Flash-attention matching for TPU: scalar epilogue leaves, no extra scalars, static offsets.
impl Attentions for TpuModule<'_> {
    fn contraction_epilogues(&self) -> bool {
        self.options.config.tpu.contraction_epilogues
    }

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

    fn runtime_offsets(&self) -> bool {
        false
    }
}

/// Lets dot merging record its packed weight blocks on the module.
impl Blocks for TpuModule<'_> {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

/// Runs the TPU pass pipeline over `graph` and plans its buffers with the TPU tiled layout.
pub(crate) fn compile(graph: &Graph, options: &Options) -> Result<Plan, String> {
    let mut module = TpuModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        interleaved: Vec::new(),
    };
    let mut pipeline = Pipeline::new("tpu");
    if options.config.tpu.merge_dots {
        pipeline.add(MergeDotsBackward);
    }
    if options.config.tpu.flash_attention {
        pipeline.add(FlashAttention).add(FlashAttentionBackward);
    }
    if options.config.tpu.safe_kernels {
        pipeline.add(ClampIndices);
    }
    pipeline
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .add(DotStrength);
    if options.config.tpu.merge_dots {
        pipeline.add(MergeDots {
            target: Target {
                reads: |_, _| true,
                views: false,
                lanes: LANES,
                columns: 1,
            },
        });
    }
    if options.config.tpu.normalization_diamonds {
        pipeline.add(Diamonds).add(DotRows);
    }
    pipeline
        .add(Updates {
            least: 1,
            most: usize::MAX,
        })
        .add(FuseDots)
        .add(PlaceConcats)
        .add(LoopFusion);
    if options.config.tpu.horizontal_fusion {
        pipeline.add(HorizontalFusion);
    }
    pipeline.run(&mut module)?;
    let TpuModule {
        graph,
        packed,
        interleaved,
        ..
    } = module;
    // Packed weight blocks are extra inputs and count as parameters.
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
        memory_limit: options.config.tpu.memory_limit(),
        // Packed blocks were appended as the last graph inputs, in order.
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
