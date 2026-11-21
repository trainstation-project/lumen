//! What the TPU code generators share: the backend type, which primitives a plain elementwise kernel runs, the JSON
//! nodes the Pallas programs rebuild primitives from, and outputs written into windows of wider matrices.

use crate::Scalar;
use crate::graph::{Primitive, UnaryOp};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::{plan_options, positions};
use crate::plan::{CpuPlan, DeviceLayout, Plan, PlanOptions};

#[cfg(lumen_tpu_linked)]
mod kernel;
#[cfg(lumen_tpu_linked)]
pub(crate) mod lowering;

#[cfg(lumen_tpu_linked)]
pub(crate) use kernel::*;

#[cfg(lumen_tpu_linked)]
use crate::plan::StepBase;

/// The TPU code generator.
pub(crate) struct Tpu;

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
        return Ok(Box::new(crate::codegen::utils::tpu::codegen(
            plan, options,
        )?));

        #[cfg(not(lumen_tpu_linked))]
        Ok(Box::new(plan))
    }
}

/// What TPU kernels run: no float16 (Mosaic's TPU kernels have no float16), and no cumsum, scatter-add or random
/// bits (no TPU kernel computes them yet).
impl crate::codegen::Supports for Tpu {
    /// Rejects graphs with a float16 value or an op no TPU kernel computes.
    fn check(graph: &crate::graph::Graph) -> Result<(), String> {
        use crate::DType;

        for node in graph.nodes() {
            let half = node
                .inputs
                .iter()
                .chain([&node.output])
                .any(|&v| graph.type_of(v).dtype == DType::F16);

            if half {
                return Err(format!(
                    "{}: float16 does not run on TPU yet",
                    node.primitive.name()
                ));
            }

            if matches!(
                node.primitive,
                Primitive::Cumsum { .. }
                    | Primitive::ScatterAdd { .. }
                    | Primitive::RandomBits { .. }
            ) {
                return Err(format!("{} does not run on TPU yet", node.primitive.name()));
            }
        }

        Ok(())
    }
}
