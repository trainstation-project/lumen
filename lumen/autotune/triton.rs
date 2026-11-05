//! Triton autotuning: the candidate configs of attention, GEMM and loop kernels, timed with CUDA events.

use super::{Kernels, Timer};
use crate::codegen::attention::triton::launch::{attention, is_attention};
use crate::codegen::attention::triton::{ATTENTION_CONFIGS, AttentionConfig};
use crate::codegen::dot_general::triton::launch::{DOT_ROW, Run, gemm, is_gemm};
use crate::codegen::dot_general::triton::{self as codegen, CONFIGS, Config};
use crate::codegen::elementwise::triton::{LoopConfig, loop_configs};
use crate::codegen::triton::{Kernel, Template};
use crate::graph::{Primitive, TensorType};
use crate::plan::{Buffer, TritonStep, Tuned};
use crate::{Device, Tensor, TensorOptions};

/// Autotuning candidates for flash-attention steps.
pub(crate) struct Tiles;

impl Kernels for Tiles {
    type Step = TritonStep;
    type Config = AttentionConfig;

    /// Every attention config for an attention step, none otherwise.
    fn configs(&self, step: &TritonStep) -> Vec<AttentionConfig> {
        match is_attention(&step.primitive) {
            true => ATTENTION_CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &TritonStep, config: AttentionConfig) -> Tuned {
        let run: Run = Box::new(move |step, inputs, out, outs| {
            attention(step, inputs, out, outs, Some(&config))
        });

        Tuned::new(format!("{config:?}"), run)
    }
}

/// Autotuning candidates for GEMM steps.
pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Step = TritonStep;
    type Config = Config;

    /// Every GEMM config for a GEMM step (whole-row tiles for a row epilogue), none otherwise.
    fn configs(&self, step: &TritonStep) -> Vec<Config> {
        match &step.primitive {
            Primitive::Fusion { name, .. } if name == DOT_ROW => {
                codegen::row_gemm_configs(step.output.1.shape.last().copied().unwrap_or(1))
            }
            _ if is_gemm(step) => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &TritonStep, config: Config) -> Tuned {
        let run: Run =
            Box::new(move |step, inputs, out, outs| gemm(step, inputs, out, outs, Some(&config)));

        Tuned::new(format!("{config:?}"), run)
    }
}

/// Autotuning candidates for loop kernels, the ones codegen kept on the steps.
pub(crate) struct Loops;

/// The loop kernel codegen kept on `step`, if it has one.
fn loop_kernel(step: &TritonStep) -> Option<&Kernel> {
    step.kernel.as_ref()
}

impl Kernels for Loops {
    type Step = TritonStep;
    type Config = LoopConfig;

    /// Loop configs whose block covers the kernel's lanes and splits; none for a scatter-add.
    fn configs(&self, step: &TritonStep) -> Vec<LoopConfig> {
        let Some(kernel) = loop_kernel(step) else {
            return Vec::new();
        };

        // Timed on zeroed indices, every update would add into one position, measuring contention on it
        // rather than the kernels: a scatter-add's launch with the configs in its config.rs.
        if let Some(Template::ScatterAdd { .. }) = kernel.template {
            return Vec::new();
        }

        loop_configs(kernel.lanes.max(kernel.splits.next_power_of_two()))
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &TritonStep, config: LoopConfig) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Times steps on `device` with CUDA events.
pub(crate) struct Events {
    pub(crate) device: Device,
}

impl Timer for Events {
    type Step = TritonStep;
    type Buffers = (Vec<Tensor>, Tensor, Vec<Tensor>);

    /// The device steps run on.
    fn device(&self) -> Device {
        self.device
    }

    /// Zeroed inputs and outputs for `step`; scalars live on the CPU, and in-place ops write their operand.
    fn buffers(&self, step: &TritonStep) -> Result<Option<Self::Buffers>, String> {
        let zeros = |ty: &TensorType, device| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(device),
            )
        };

        let inputs: Vec<Tensor> = step
            .inputs
            .iter()
            .map(|(b, ty)| match b {
                Buffer::Scalar(_) => zeros(ty, Device::Cpu(0)),
                _ => zeros(ty, self.device),
            })
            .collect();

        let out = match step.primitive {
            Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. } => inputs[0].clone(),
            _ => zeros(&step.output.1, self.device),
        };

        let outs = step
            .extra_outputs
            .iter()
            .map(|(_, ty)| zeros(ty, self.device))
            .collect();

        Ok(Some((inputs, out, outs)))
    }

    /// One run of the step.
    fn run(
        &self,
        (inputs, out, outs): &Self::Buffers,
        step: &TritonStep,
        _: usize,
    ) -> Result<(), String> {
        step.run_raw(inputs, out, outs)
    }
}
