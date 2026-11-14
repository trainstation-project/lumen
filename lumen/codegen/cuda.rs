//! CUDA (CuTe DSL) codegen: each planned step's kernel, generated when the plan compiles for the layouts it
//! gives the step's tensors and kept in the step's memo; tensors laid out otherwise get theirs at launch.

use crate::autotune::Autotuner;
use crate::autotune::cuda::{AttentionForwards, Events, Gemms};
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::dot_general::cuda::matmul;
use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::cuda::Layouts;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::utils::cuda::Cuda;
use crate::graph::{Graph, Primitive, TensorType};
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::attention::FORWARD;
use crate::graph_compiler::{Options, plan_options};
use crate::plan::{CpuPlan, CudaPlan, CudaStep, Plan, Step as _, StepBase};
use std::sync::Arc;

/// The kernel `step` runs as for tensors laid out as `layouts` (a GEMM fusion's epilogue), generated once per layout
/// and kept in the step's memo.
pub(crate) fn generated(
    step: &StepBase,
    layouts: Layouts,
) -> Result<Arc<(String, String)>, String> {
    let key = layouts
        .inputs
        .iter()
        .chain(&layouts.outputs)
        .chain(std::iter::once(&layouts.spans))
        .fold(0u64, |h, s| {
            use std::hash::{Hash, Hasher};

            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (h, s).hash(&mut hasher);
            hasher.finish()
        });

    step.memo.get(key, || {
        let by_value = vec![false; layouts.inputs.len()];

        let body = match &step.primitive {
            Primitive::Fusion { body, .. } => match gemm_dot(body) {
                Some(dot) => return Cuda::gemm(body, dot, &by_value, &layouts),
                None => body.clone(),
            },
            Primitive::Reshape { new_sizes } => copy_body(&step.inputs[0].1, new_sizes),
            p => {
                return Err(format!(
                    "{}: the CUDA backend has no kernel for it yet",
                    p.name()
                ));
            }
        };

        Cuda::elementwise(&body, &by_value, &Default::default(), &layouts)
    })
}

/// Elements a tensor of `shape` laid out at `strides` spans, from its first to its last.
fn extent(shape: &[usize], strides: &[usize]) -> usize {
    match shape.contains(&0) {
        true => 0,
        false => {
            shape
                .iter()
                .zip(strides)
                .map(|(&n, &s)| (n - 1) * s)
                .sum::<usize>()
                + 1
        }
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

    plan.map(|step| Ok(CudaStep(step.into_base())))
}

/// The body of a copy of `x` reshaped to `shape`: its input, reshaped, its output.
fn copy_body(x: &TensorType, shape: &[usize]) -> Graph {
    let mut body = Graph::new();
    let input = body.input(x.clone());

    let output = body
        .apply(
            Primitive::Reshape {
                new_sizes: shape.to_vec(),
            },
            &[input],
        )
        .expect("a reshape of the copied value");

    body.set_outputs(&[output]).expect("the copied value");
    body
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
        let mut plan = crate::codegen::cuda::codegen(plan, options)?;

        if config.autotune || crate::autotune::forcing() {
            let tuner = Autotuner {
                timer: Events { device },
                autotune: config.autotune,
            };

            tuner.tune(&mut Gemms, &mut plan)?;
            tuner.tune(&mut AttentionForwards, &mut plan)?;
        }

        Ok(Box::new(plan))
    }
}
