//! Running CUDA (CuTe DSL) steps: each step's kernel, generated for its tensors' layouts.

use std::ops::{Deref, DerefMut};

use super::triton::traced;
use super::{unsupported, viewed};
use crate::codegen::dot_general::cuda::Config;
use crate::codegen::elementwise::cuda::Layouts;
use crate::graph::Primitive;
use crate::graph_compiler::passes::attention::FORWARD;
use crate::plan::run::Buffers;
use crate::plan::{Step, StepBase};
use crate::runtime::kernel::cuda::{GEMM, LAUNCH};
use crate::{Device, Tensor};

/// A step of a CUDA (CuTe DSL) kernel, generated for its tensors' layouts and kept in its memo.
#[derive(Debug, Clone)]
pub struct CudaStep(pub StepBase);

impl Deref for CudaStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.0
    }
}

impl DerefMut for CudaStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.0
    }
}

impl Step for CudaStep {
    /// Runs the step on its tensors through the kernel generated for their layouts.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let (inputs, out, outs) = buffers.tensors(self);
        traced(self.label, &out, || self.run_raw(&inputs, &out, &outs))
    }

    /// Errors unless `device` is a CUDA device and no step reads an operand through a view.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        let runs = |d| matches!(d, Device::Cuda(_));

        match unsupported(steps, device, runs, "CUDA").or_else(|| viewed(steps)) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step itself.
    fn into_base(self) -> StepBase {
        self.0
    }
}

impl CudaStep {
    /// Runs the step on the given tensors: a matmul through the GEMM (with its tuned tile, if autotuned), a flash
    /// attention through the attention forward (with its tuned config, if autotuned), or an elementwise fusion or an output's copy (a reshape) through the kernel generated for the tensors' layouts (kept in
    /// the step's memo), launched by Python.
    pub(crate) fn run_raw(
        &self,
        inputs: &[Tensor],
        out: &Tensor,
        outs: &[Tensor],
    ) -> Result<(), String> {
        let step = &self.0;

        // A matmul runs on the GEMM, A as (.., m, k) and B as (.., k, n) whichever of their dims contract.
        if let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            ..
        } = &step.primitive
        {
            let gemm = GEMM.get().ok_or(
                "the CUDA backend's GEMM runs from Python (lumen.codegen.dot_general.cuda): none is installed",
            )?;

            if out.numel() == 0 {
                return Ok(());
            }

            let rank = inputs[0].ndim();

            let a = match lhs_contracting[0] == rank - 1 {
                true => inputs[0].clone(),
                false => inputs[0].transpose(rank - 2, rank - 1),
            };

            let b = match rhs_contracting[0] == rank - 2 {
                true => inputs[1].clone(),
                false => inputs[1].transpose(rank - 2, rank - 1),
            };

            return gemm(&a, &b, out, step.launch::<Config>());
        }

        if let Primitive::Fusion { name, .. } = &step.primitive
            && name == FORWARD
        {
            return crate::codegen::attention::cuda::run(step, inputs, out, outs, step.launch());
        }

        let launch = LAUNCH.get().ok_or(
            "the CUDA backend launches from Python (lumen.codegen.utils.cuda): none is installed",
        )?;

        let outputs: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
        let n = outputs.iter().map(|t| t.numel()).max().unwrap_or(0);

        if n == 0 {
            return Ok(());
        }

        let tensors: Vec<Tensor> = inputs
            .iter()
            .chain(outputs.iter().copied())
            .map(Tensor::flat)
            .collect();

        let layouts = Layouts {
            inputs: inputs.iter().map(|t| t.strides().to_vec()).collect(),
            outputs: outputs.iter().map(|t| t.strides().to_vec()).collect(),
            spans: tensors.iter().map(Tensor::numel).collect(),
        };

        let code = crate::codegen::cuda::generated(step, layouts)?;
        let (name, source) = code.as_ref();
        launch(name, source, &tensors, outputs.len(), n)
    }
}
