//! CUDA (CuTe DSL) codegen: each planned step's kernel, generated when the plan compiles for the layouts it
//! gives the step's tensors and kept in the step's memo; tensors laid out otherwise get theirs at launch.

use std::sync::Arc;

use crate::codegen::elementwise::ElementwiseCodegen;
use crate::codegen::elementwise::cuda::Layouts;
use crate::codegen::utils::cuda::Cuda;
use crate::graph::{Graph, Primitive, TensorType};
use crate::graph_compiler::Options;
use crate::plan::{CpuPlan, CudaPlan, CudaStep, Step as _, StepBase};

/// The kernel `step` runs as for tensors laid out as `layouts`, generated once per layout and kept in the
/// step's memo.
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
        let body = match &step.primitive {
            Primitive::Fusion { body, .. } => body.clone(),
            Primitive::Reshape { new_sizes } => copy_body(&step.inputs[0].1, new_sizes),
            p => {
                return Err(format!(
                    "{}: the CUDA backend has no kernel for it yet",
                    p.name()
                ));
            }
        };

        let by_value = vec![false; layouts.inputs.len()];
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

        // A matmul runs on the GEMM, compiled at its first launch for its operands' majors.
        if outputs.clone().all(|(_, ty)| ty.numel() == 0)
            || matches!(step.primitive, Primitive::DotGeneral { .. })
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
