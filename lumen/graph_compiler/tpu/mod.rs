//! TPU graph compiler: the pass pipeline that rewrites a traced graph into TPU kernels and plans its buffers.
#[cfg(test)]
mod tests;

use super::common::dot::rows::tpu::DotRows;
use super::common::dot::tpu::FuseDots;
use super::common::fusion::tpu::{self as fusion, Fusion};
use super::common::updates::Updates;
use crate::graph_compiler::common::diamonds::FusedDiamonds;
use crate::graph_compiler::common::dot::canonicalize::CanonicalizeDots;
use crate::graph_compiler::common::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::common::place_concats::{PlaceConcats, Slots};

use super::common::attention::tpu::AttentionLayouts;
use super::common::attention::{FlashAttention, FlashAttentionBackward};
use super::common::clamp_indices::ClampIndices;
use super::common::dot::fold_reshapes::FoldDotReshapes;
use super::common::dot::strength::DotStrength;
use super::common::merge_dots::{Blocks, MergeDots, MergeDotsBackward, Packs, Target};
use super::common::simplify::Simplify;
use super::pass::{Module, Pipeline};
use super::{CompilerConfig, Options, plan_options, positions};
use crate::graph::{Graph, Node, Primitive, TensorType};
use crate::graph_compiler::{Plan, PlanOptions};
use crate::plan::DeviceLayout;

/// Vector lane width of a TPU register; concatenated and merged dot widths are aligned to it.
pub(crate) const LANES: usize = 128;

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
pub(crate) struct TpuModule<'a> {
    pub(crate) graph: Graph,
    pub(crate) options: &'a Options,
    /// Packed weight blocks built by dot merging, appended as the graph's last inputs.
    packed: Packs,
    /// Per-pack region layout of interleaved members, parallel to `packed`.
    interleaved: Vec<Vec<usize>>,
}

/// Exposes the graph and the TPU compiler config to the shared pass framework.
impl Slots for TpuModule<'_> {
    /// What `fusion::slot_writes` accepts.
    fn writes(&self, concat: &Node, k: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, concat, k, producer)
    }
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

    fn config(&self) -> &CompilerConfig {
        &self.options.config
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

    if options.config.merge_dots {
        pipeline.add(MergeDotsBackward);
    }

    if options.config.flash_attention {
        pipeline
            .add(FlashAttention)
            .add(FlashAttentionBackward)
            .add(AttentionLayouts);
    }

    if options.config.safe_kernels {
        pipeline.add(ClampIndices);
    }

    pipeline
        .add(FoldDotReshapes)
        .add(Simplify { matched: true })
        .add(DotStrength);

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

    // Matmul operands the TPU kernel can't read as laid out are transposed into its form first.
    pipeline.add(CanonicalizeDots {
        readable: reads_as_laid_out,
    });

    if options.config.normalization_diamonds {
        pipeline
            .add(FusedDiamonds {
                fusible: crate::codegen::row::tpu::fusible,
                fits: crate::codegen::row::tpu::fits,
            })
            .add(DotRows);
    }

    pipeline
        .add(Updates {
            least: 1,
            most: usize::MAX,
        })
        .add(FuseDots)
        .add(PlaceConcats)
        .add(Fusion);

    if options.config.horizontal_fusion {
        pipeline.add(HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Shape,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        });
    }

    pipeline.run(&mut module)?;

    let TpuModule {
        graph,
        packed,
        interleaved,
        ..
    } = module;

    let config = &options.config;

    let plan = PlanOptions {
        layout: Some(LAYOUT),
        ..plan_options(
            &graph,
            options,
            config,
            &packed,
            positions(&options.scalars),
        )
    };

    let mut plan = Plan::compile_with(&graph, &plan);
    plan.packed = packed;
    plan.interleaved = interleaved;

    // Each step's kernel is generated now, so running the plan only launches them.
    #[cfg(lumen_tpu_linked)]
    crate::runtime::tpu::bake(&plan);

    Ok(plan)
}

/// Whether the TPU matmul kernel reads operand `k` of `dot` over `operands` as laid out: its batch dimensions
/// lead it, and the lhs is batch, free dimensions, then one contracting one (or batch, contracting, free), the
/// rhs batch, then one contracting and one free dimension in either order.
fn reads_as_laid_out(dot: &Primitive, operands: [&TensorType; 2], k: usize) -> bool {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = dot
    else {
        unreachable!("a dot_general")
    };

    let leads = |batch: &[usize]| batch.iter().enumerate().all(|(i, &d)| d == i);
    let nb = lhs_batch.len();
    let rank = operands[k].shape.len();

    match k {
        0 => {
            let last = *lhs_contracting == [rank - 1];
            let transposed = rank == nb + 2 && *lhs_contracting == [nb];
            leads(lhs_batch) && rank >= nb + 2 && (last || transposed)
        }
        _ => leads(rhs_batch) && rank == nb + 2 && rhs_contracting.len() == 1,
    }
}
