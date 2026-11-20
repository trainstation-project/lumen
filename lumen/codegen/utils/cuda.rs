//! What the CUDA (CuTe DSL) code generators share: the backend type, element types and constants.
//!
//! CUDA (CuTe DSL) codegen: each planned step's kernel, generated when the plan compiles for the layouts it
//! gives the step's tensors and kept in the step's memo; tensors laid out otherwise get theirs at launch.

use crate::autotune::Autotuner;
use crate::codegen::attention::cuda::FlashAttention;
use crate::codegen::dot_general::cuda::Gemm;
use crate::codegen::dot_general::cuda::matmul;
use crate::codegen::elementwise::cuda::{Elementwise, Layouts};
use crate::codegen::elementwise::cuda::{extent, generated};
use crate::codegen::fusion::cuda::{ELEMENTWISE, fusible};
use crate::codegen::kernel::Runner;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::utils::triton::bits;
use crate::graph::{Graph, Node, Primitive};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::attention::FORWARD;
use crate::graph_compiler::{Options, plan_options};
use crate::plan::step::Tensors;
use crate::plan::step::cuda::CudaTimer;
use crate::plan::{CpuPlan, CudaPlan, CudaStep, Plan, Step as _};
use crate::{DType, Scalar};
use std::sync::Arc;

/// The CUDA code generator: kernels in NVIDIA's CuTe DSL (`cutlass.cute`), compiled at launch.
pub(crate) struct Cuda;

/// The CuTe numeric type of `dtype`, or `None` for one it has no kernels for.
pub(crate) fn cute_type(dtype: DType) -> Option<&'static str> {
    use DType::*;

    Some(match dtype {
        Bool => "cutlass.Boolean",
        U8 => "cutlass.Uint8",
        U16 => "cutlass.Uint16",
        U32 => "cutlass.Uint32",
        U64 => "cutlass.Uint64",
        I8 => "cutlass.Int8",
        I16 => "cutlass.Int16",
        I32 => "cutlass.Int32",
        I64 => "cutlass.Int64",
        F16 => "cutlass.Float16",
        BF16 => "cutlass.BFloat16",
        F32 => "cutlass.Float32",
        F64 => "cutlass.Float64",
        _ => return None,
    })
}

/// The unsigned CuTe type of `bits` bits.
pub(crate) fn unsigned(bits: usize) -> &'static str {
    match bits {
        8 => "cutlass.Uint8",
        16 => "cutlass.Uint16",
        32 => "cutlass.Uint32",
        _ => "cutlass.Uint64",
    }
}

/// `value` as a CuTe constant of `dtype`, written as its bits so every dtype's value is exact.
pub(crate) fn constant(dtype: DType, value: Scalar) -> String {
    let bits = bits(dtype, value);
    let t = cute_type(dtype).expect("a CuTe dtype");

    match dtype {
        DType::Bool => format!("cutlass.Boolean({})", bits != 0),
        d if d.is_float() => format!("{}({bits:#x}).bitcast({t})", unsigned(d.bits())),
        _ => format!("{t}({})", signed_bits(dtype, bits)),
    }
}

/// The integer `dtype` holds as `bits`, written as a Python literal.
fn signed_bits(dtype: DType, bits: u64) -> String {
    let width = dtype.bits();

    match dtype {
        DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            let shifted = (bits << (64 - width)) as i64 >> (64 - width);
            shifted.to_string()
        }
        _ => bits.to_string(),
    }
}

/// The CUDA backend's plan of `plan`: each step's kernel generated for the layouts the plan gives its tensors,
/// kept in the step's memo by layout so its launches find it; tensors laid out otherwise when it runs get theirs
/// then. Fails on a step no CUDA kernel takes.
pub(crate) fn codegen(plan: CpuPlan, _: &Options) -> Result<CudaPlan, String> {
    for step in plan.steps() {
        let outputs = std::iter::once(&step.output).chain(&step.extra_outputs);

        // A matmul runs on the GEMM and a flash attention on the attention forward, each compiled at its first
        // launch for its operands' layouts.
        let attention =
            matches!(&step.primitive, Primitive::Fusion { name, .. } if name == FORWARD);

        if outputs.clone().all(|(_, ty)| ty.numel() == 0)
            || matches!(step.primitive, Primitive::DotGeneral { .. })
            || attention
        {
            continue;
        }

        let (inputs, outputs_at) = (step.input_strides(), step.output_strides());
        let types = step.inputs.iter().chain(outputs).map(|(_, ty)| &ty.shape);

        let spans = types
            .zip(inputs.iter().chain(&outputs_at))
            .map(|(shape, strides)| extent(shape, strides))
            .collect();

        let layouts = Layouts {
            inputs,
            outputs: outputs_at,
            spans,
        };

        generated(step, layouts)?;
    }

    plan.map(|step| {
        let base = step.into_base();

        // Each step's kernel: the GEMM for a matmul or a GEMM fusion, the attention forward for a flash attention,
        // else the kernel generated for its elementwise body (or its copy).
        let kernel: Arc<dyn Runner<CudaStep, Tensors>> = match &base.primitive {
            Primitive::DotGeneral { .. } => Arc::new(Gemm),
            Primitive::Fusion { name, .. } if name == FORWARD => Arc::new(FlashAttention),
            Primitive::Fusion { body, .. } if gemm_dot(body).is_some() => Arc::new(Gemm),
            _ => Arc::new(Elementwise),
        };

        Ok(CudaStep { base, kernel })
    })
}

/// The CUDA backend plans its fused graph, every runtime scalar a 0-d device input, and autotunes its GEMM tiles
/// and attention configs when enabled; it errors on an op it has no kernel for.
impl Device for Cuda {
    const NAME: &'static str = "cuda";

    /// Plans `module`'s graph, generates its kernels and autotunes them.
    fn lower(
        module: DeviceModule<'_, Self>,
        device: crate::Device,
    ) -> Result<Box<dyn Plan>, String> {
        let DeviceModule {
            graph: fused,
            options,
            packed,
            interleaved,
            ..
        } = module;

        if let Some(node) = fused.nodes().iter().find(|n| {
            !matches!(
                n.primitive,
                Primitive::Fusion { .. }
                    | Primitive::FusionOutput { .. }
                    | Primitive::Reshape { .. }
            ) && !matmul(&fused, n)
        }) {
            return Err(format!(
                "{}: the CUDA (CuTe DSL) backend has no kernel for it yet; turn \
                 config.compiler.use_triton_backend on to run it on Triton",
                node.primitive.name()
            ));
        }

        let config = &options.config;
        let plan = plan_options(&fused, options, config, &packed, Vec::new());
        let mut plan = CpuPlan::compile_with(&fused, &plan);
        plan.packed = packed;
        plan.interleaved = interleaved;
        let mut plan = crate::codegen::utils::cuda::codegen(plan, options)?;

        if config.autotune || crate::autotune::forcing() {
            let tuner = Autotuner {
                timer: CudaTimer { device },
                autotune: config.autotune,
            };

            tuner.tune_steps(&mut plan)?;
        }

        Ok(Box::new(plan))
    }
}

impl crate::codegen::Lone for DeviceModule<'_, Cuda> {
    /// Whether lone `n` needs a one-node kernel: every fusible node but reshapes, which are views.
    fn alone(graph: &Graph, n: &Node) -> bool {
        fusible(graph, n) && !matches!(n.primitive, Primitive::Reshape { .. })
    }

    /// The CUDA backend passes no inputs by value: runtime scalars are 0-d device tensors.
    fn scalars(&mut self) -> Option<&mut Vec<usize>> {
        None
    }

    /// Every kernel is an elementwise one.
    fn kernel(&mut self, _: &Graph, _: &[bool]) -> String {
        ELEMENTWISE.into()
    }
}
