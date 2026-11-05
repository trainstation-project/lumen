//! The Triton backend for CUDA: its passes, plan compilation and autotuning; `runtime::triton` runs the steps
//! through Triton kernels.

use super::passes::Module;
use super::passes::alone::Alone;
use super::passes::attention::{FlashAttention, FlashAttentionBackward};
use super::passes::chunked_cross_entropy::ChunkedCrossEntropy;
use super::passes::clamp_indices::ClampIndices;
use super::passes::dot::rows::triton::WideRows;
use super::passes::dot::split_k::triton::SplitK;
use super::passes::dot::strength::DotStrength;
use super::passes::fusion::triton::{self as fusion, Fusion};
use super::passes::linear_cross_entropy::LinearCrossEntropy;
use super::passes::linear_reordering::LinearReordering;
use super::passes::merge_dots::{Blocks, MergeDots, MergeDotsBackward, Packs, Target};
use super::passes::simplify::Simplify;
use super::passes::take_along_axis::TakeAlongAxis;
use super::passes::updates::Updates;
use super::pipeline::{Pass, Passes, pipeline, when};
use super::{CompilerConfig, Options, plan_options, positions};
use crate::autotune::Autotuner;
use crate::autotune::triton::{Events, Gemms, Loops, Tiles};
use crate::graph::{Graph, Node};
use crate::graph_compiler::passes::diamonds::{Diamonds, Row, RowsModule};
use crate::graph_compiler::passes::horizontal::{Grouping, HorizontalFusion};
use crate::graph_compiler::passes::place_concats::{PlaceConcats, Slots};
use crate::graph_compiler::passes::place_views::PlaceViews;
use crate::plan::{CpuPlan, TritonPlan};
use crate::{DType, Device};

/// Rejects graphs with half-precision transcendentals, which Triton's libdevice lacks.
fn unsupported(graph: &Graph) -> Result<(), String> {
    for node in graph.nodes() {
        let x = graph.type_of(node.inputs.first().copied().unwrap_or(node.output));

        if node.primitive.is_transcendental() && matches!(x.dtype, DType::F16 | DType::BF16) {
            return Err(format!(
                "{}: has no {} kernel on Triton, got {x}: convert to float32 first (x.to(dtype=float32))",
                node.primitive.name(),
                x.dtype.name()
            ));
        }
    }

    Ok(())
}

/// A graph compiled for Triton, with the packs merge_dots creates.
pub(crate) struct TritonModule<'a> {
    pub(crate) graph: Graph,
    pub(crate) options: &'a Options,
    /// Inputs concatenated into each packed block.
    packed: Packs,
    /// Interleaved region sizes of each packed block.
    interleaved: Vec<Vec<usize>>,
    /// Normalization rows the diamonds pass found, consumed by fusion.
    pub(crate) rows: Vec<Row>,
}

impl Module for TritonModule<'_> {
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

impl Slots for TritonModule<'_> {
    /// Kernels of their own writing fresh memory (`fusion::slot_writes`).
    fn writes(&self, _: &Node, _: usize, producer: &Node, _: usize) -> bool {
        fusion::slot_writes(&self.graph, producer)
    }
}

impl RowsModule for TritonModule<'_> {
    fn rows_mut(&mut self) -> &mut Vec<Row> {
        &mut self.rows
    }
}

impl Blocks for TritonModule<'_> {
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

/// Triton's passes, with optional ones enabled by `config`; with fusion off (`fuse`), as on MPS, no pass merges
/// ops and each runs as its own kernel.
impl Passes for TritonModule<'_> {
    const NAME: &'static str = "triton";

    /// Always.
    fn take_along_axis(_: &CompilerConfig) -> Pass<Self> {
        Box::new(TakeAlongAxis)
    }

    /// With `fuse` and `fused_linear_cross_entropy`.
    fn chunked_cross_entropy(config: &CompilerConfig) -> Pass<Self> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            ChunkedCrossEntropy,
        )
    }

    /// With `fuse` and `merge_dots`.
    fn merge_dots_backward(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse && config.merge_dots, MergeDotsBackward)
    }

    /// With `fuse` and `merge_dots`.
    fn merge_dots(config: &CompilerConfig) -> Pass<Self> {
        let merge = MergeDots {
            target: Target {
                reads: fusion::fusible,
                views: true,
                lanes: 1,
                columns: 8,
            },
        };

        when(config.fuse && config.merge_dots, merge)
    }

    /// With `fuse` and `flash_attention`.
    fn flash_attention(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse && config.flash_attention, FlashAttention)
    }

    /// With `fuse` and `flash_attention`.
    fn flash_attention_backward(config: &CompilerConfig) -> Pass<Self> {
        when(
            config.fuse && config.flash_attention,
            FlashAttentionBackward,
        )
    }

    /// With `fuse` and `fused_linear_cross_entropy`.
    fn linear_cross_entropy(config: &CompilerConfig) -> Pass<Self> {
        when(
            config.fuse && config.fused_linear_cross_entropy,
            LinearCrossEntropy,
        )
    }

    /// With `safe_kernels`.
    fn clamp_indices(config: &CompilerConfig) -> Pass<Self> {
        when(config.safe_kernels, ClampIndices)
    }

    /// Always.
    fn simplify(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Simplify { matched: true })
    }

    /// Always: batched inner products (attention backward's delta, read through transposes) run as elementwise
    /// fusions.
    fn dot_strength(_: &CompilerConfig) -> Pass<Self> {
        Box::new(DotStrength)
    }

    /// With `linear_reordering`, as on MPS: a matmul operand's factor along the kept axes moves to its output.
    fn linear_reordering(config: &CompilerConfig) -> Pass<Self> {
        let reordering = LinearReordering {
            tile: Some(crate::codegen::dot_general::triton::MOST_GEMM_ROW),
        };

        when(config.linear_reordering, reordering)
    }

    /// With `split_k`.
    fn split_k(config: &CompilerConfig) -> Pass<Self> {
        when(config.split_k, SplitK)
    }

    /// With `fuse`.
    fn updates(config: &CompilerConfig) -> Pass<Self> {
        let updates = Updates {
            least: 2,
            most: usize::MAX,
        };

        when(config.fuse, updates)
    }

    /// With `fuse` and `normalization_diamonds`: normalization rows for the fusion, as MPS finds them.
    fn diamonds(config: &CompilerConfig) -> Pass<Self> {
        let diamonds = Diamonds {
            fusible: fusion::fusible,
            partials: crate::codegen::row::triton::ROW_PARTIALS,
        };

        when(config.fuse && config.normalization_diamonds, diamonds)
    }

    /// With `fuse`.
    fn place_concats(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse, PlaceConcats)
    }

    /// With `fuse`.
    fn fusion(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse, Fusion)
    }

    /// With `fuse` and `normalization_diamonds`: rows too wide for one GEMM tile get the GEMM feeding them
    /// writing their reductions' partials too.
    fn wide_rows(config: &CompilerConfig) -> Pass<Self> {
        when(config.fuse && config.normalization_diamonds, WideRows)
    }

    /// Always: Triton's GEMM reads its operands at any strides.
    fn place_views(_: &CompilerConfig) -> Pass<Self> {
        Box::new(PlaceViews {
            reads: |_, _, _, _| true,
        })
    }

    /// Always.
    fn alone(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Alone)
    }

    /// With `fuse` and `horizontal_fusion`.
    fn horizontal_fusion(config: &CompilerConfig) -> Pass<Self> {
        let horizontal = HorizontalFusion {
            candidate: fusion::horizontal_candidate,
            grouping: Grouping::Tiling,
            buffers: fusion::HORIZONTAL_BUFFERS,
            name: "elementwise",
        };

        when(config.fuse && config.horizontal_fusion, horizontal)
    }
}

/// Compiles `graph` for `device` with the Triton backend: runs the passes, plans the steps, generates
/// loop kernels and autotunes GEMM and loop configs when enabled.
pub(crate) fn compile(
    graph: &Graph,
    device: Device,
    options: &Options,
) -> Result<TritonPlan, String> {
    unsupported(graph)?;
    let config = &options.config;

    let mut module = TritonModule {
        graph: graph.clone(),
        options,
        packed: Vec::new(),
        interleaved: Vec::new(),
        rows: Vec::new(),
    };

    pipeline::<TritonModule>(config).run(&mut module)?;

    let TritonModule {
        graph: fused,
        packed,
        interleaved,
        ..
    } = module;

    let scalars = positions(&options.scalars);
    let plan = plan_options(&fused, options, config, &packed, scalars);
    let mut plan = CpuPlan::compile_with(&fused, &plan);
    plan.packed = packed;
    plan.interleaved = interleaved;
    let mut plan = crate::codegen::triton::codegen(plan, options)?;

    if options.config.autotune {
        let tuner = Autotuner {
            timer: Events { device },
        };

        tuner.tune(&mut Gemms, &mut plan)?;
        tuner.tune(&mut Tiles, &mut plan)?;
        tuner.tune(&mut Loops, &mut plan)?;
    }

    Ok(plan)
}
