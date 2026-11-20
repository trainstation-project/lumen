//! What the TPU code generators share: the backend type, which primitives a plain elementwise kernel runs, the JSON
//! nodes the Pallas programs rebuild primitives from, and outputs written into windows of wider matrices.

use crate::Scalar;
use crate::graph::{Primitive, UnaryOp};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::{plan_options, positions};
use crate::plan::{CpuPlan, DeviceLayout, Plan, PlanOptions};

#[cfg(lumen_tpu_linked)]
pub(crate) mod lowering;

#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;

/// The TPU code generator.
pub(crate) struct Tpu;

/// TPU fuses chains of 1 or more updates, always: even one update runs as an `updates` kernel. Here, where `Tpu`
/// is built without the TPU linked too, so every build's TPU pipeline has the pass (as its tests check).
impl crate::codegen::dynamic_slice::UpdateChains for Tpu {
    const LEAST: usize = 1;
    const MOST: usize = usize::MAX;
    const WITH_FUSE: bool = false;
}

/// TPU's kernels writing a concatenation's operands into their slots, always (here, where `Tpu` is built without the
/// TPU linked too): what `fusion::tpu::slot_writes` accepts, every operand worth placing.
impl crate::codegen::Slots for DeviceModule<'_, Tpu> {
    const WITH_FUSE: bool = false;

    /// What `fusion::tpu::slot_writes` accepts.
    fn writes(
        &self,
        concat: &crate::graph::Node,
        k: usize,
        producer: &crate::graph::Node,
        _: usize,
    ) -> bool {
        crate::graph_compiler::passes::fusion::tpu::slot_writes(&self.graph, concat, k, producer)
    }

    /// Any operand: on TPU's tiled layout a concatenation fusion would build is a copy of its own.
    fn worth(&self, _: &crate::graph::Node, _: usize, _: &crate::graph::Node, _: usize) -> bool {
        true
    }
}

/// TPU merges without `fuse`, late (once the graph is simplified), packing whole vector lanes: everything read in place,
/// no views, `LANES` lanes, 1 column. Here, where `Tpu` is built without the TPU linked too.
impl crate::codegen::MergesDots for Tpu {
    const WITH_FUSE: bool = false;
    const LATE: bool = true;

    /// Everything read in place; no views; `LANES` lanes; 1 column.
    fn target() -> crate::graph_compiler::passes::merge_dots::Target {
        crate::graph_compiler::passes::merge_dots::Target {
            reads: |_, _| true,
            views: false,
            lanes: LANES,
            columns: 1,
        }
    }
}

/// TPU merges non-scalar `elementwise` fusions grouped by output shape, without `fuse` too. Here, where `Tpu` is built
/// without the TPU linked too.
impl crate::codegen::HorizontallyFuses for Tpu {
    const WITH_FUSE: bool = false;
    const GROUPING: crate::graph_compiler::passes::horizontal::Grouping =
        crate::graph_compiler::passes::horizontal::Grouping::Shape;
    const BUFFERS: usize = crate::graph_compiler::passes::fusion::tpu::HORIZONTAL_BUFFERS;

    /// `fusion::tpu::horizontal_candidate`.
    fn candidate(graph: &crate::graph::Graph, node: &crate::graph::Node) -> bool {
        crate::graph_compiler::passes::fusion::tpu::horizontal_candidate(graph, node)
    }
}

/// Whether `p` can run inside a plain elementwise TPU kernel (only scalar broadcasts among layout ops).
pub(crate) fn fusible(p: &Primitive) -> bool {
    use Primitive::*;

    if let BroadcastInDim {
        broadcast_dimensions,
        ..
    } = p
    {
        return broadcast_dimensions.is_empty();
    }
    matches!(
        p,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Full { .. }
            | Select
            | Unary(_)
            | Binary(_)
    )
}

/// The JSON node `{"name", "params", "args"}` the Python kernels rebuild `p` from, with `args` indexing earlier values.
pub(crate) fn node(p: &Primitive, args: &[usize]) -> String {
    let params = match p {
        Primitive::Full {
            fill_value, dtype, ..
        } => {
            // JSON has no NaN or infinity literals; Python's json module reads these spellings.
            let value = match *fill_value {
                Scalar::Bool(b) => b.to_string(),
                Scalar::Int(i) => i.to_string(),
                Scalar::Float(f) if f.is_nan() => "NaN".into(),
                Scalar::Float(f) if f.is_infinite() => {
                    format!("{}Infinity", if f < 0.0 { "-" } else { "" })
                }
                Scalar::Float(f) => format!("{f:?}"),
            };

            format!(
                "\"fill_value\": {value}, \"dtype\": \"{}\"",
                dtype.full_name()
            )
        }
        Primitive::BroadcastInDim { .. } => "\"broadcast_dimensions\": []".into(),
        Primitive::Cast { new_dtype } => format!("\"new_dtype\": \"{}\"", new_dtype.full_name()),
        Primitive::Unary(UnaryOp::ReducePrecision {
            exponent_bits,
            mantissa_bits,
        }) => format!("\"exponent_bits\": {exponent_bits}, \"mantissa_bits\": {mantissa_bits}"),
        _ => String::new(),
    };

    format!(
        "{{\"name\": \"{}\", \"params\": {{{params}}}, \"args\": {args:?}}}",
        p.name()
    )
}

/// The `(row stride, column offset)` of output `k` when it is written into a window of a wider matrix.
#[cfg(lumen_tpu_linked)]
pub(crate) fn window(step: &StepBase, k: usize) -> Option<(usize, usize)> {
    let view = step.output_views.get(k)?.as_ref()?;
    window_at(view.offset, &view.strides)
}

/// The `(row stride, column offset)` of an output written at `offset` and `strides`, when that is a
/// window of a wider matrix (rank 2 or more).
#[cfg(lumen_tpu_linked)]
pub(crate) fn window_at(offset: usize, strides: &[usize]) -> Option<(usize, usize)> {
    let rank = strides.len();
    (rank >= 2).then(|| (strides[rank - 2], offset % strides[rank - 2]))
}

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

/// TPU plans its passed graph's buffers with the TPU tiled layout and generates each step's kernel; without a TPU
/// linked, the plan stays the CPU's.
impl Device for Tpu {
    const NAME: &'static str = "tpu";

    /// Plans `module`'s graph with its packed blocks in the tiled layout, generating its kernels.
    fn lower(module: DeviceModule<'_, Self>, _: crate::Device) -> Result<Box<dyn Plan>, String> {
        let DeviceModule {
            graph,
            options,
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

        let mut plan = CpuPlan::compile_with(&graph, &plan);
        plan.packed = packed;
        plan.interleaved = interleaved;

        // Each step's kernel is generated now, so running the plan only launches them.
        #[cfg(lumen_tpu_linked)]
        return Ok(Box::new(crate::codegen::tpu::codegen(plan, options)?));

        #[cfg(not(lumen_tpu_linked))]
        Ok(Box::new(plan))
    }
}
