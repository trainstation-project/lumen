//! Triton autotuning: the candidate configs of attention, GEMM and loop kernels, timed with CUDA events.

use super::{Kernels, Timer};
use crate::codegen::attention::triton::{ATTENTION_CONFIGS, AttentionConfig};
use crate::codegen::dot_general::triton::{self as codegen, CONFIGS, Config};
use crate::codegen::elementwise::triton::{LoopConfig, loop_configs};
use crate::codegen::triton::{Kernel, Template, loop_kernel};
use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::common::attention::triton::{attention, is_attention};
use crate::graph_compiler::common::dot::triton::{DOT_ROW, Run, gemm, is_gemm};
use crate::graph_compiler::triton::run;
use crate::plan::{Buffer, Step, Tuned};
use crate::{Device, Tensor, TensorOptions};

/// Autotuning candidates for flash-attention steps.
pub(crate) struct Tiles;

impl Kernels for Tiles {
    type Config = AttentionConfig;

    /// Every attention config for an attention step, none otherwise.
    fn configs(&self, step: &Step) -> Vec<AttentionConfig> {
        match is_attention(&step.primitive) {
            true => ATTENTION_CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &Step, config: AttentionConfig) -> Tuned {
        let run: Run = Box::new(move |step, inputs, out, outs| {
            attention(step, inputs, out, outs, Some(&config))
        });

        Tuned::new(format!("{config:?}"), run)
    }
}

/// Autotuning candidates for GEMM steps.
pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Config = Config;

    /// Every GEMM config for a GEMM step (whole-row tiles for a row epilogue), none otherwise.
    fn configs(&self, step: &Step) -> Vec<Config> {
        match &step.primitive {
            Primitive::Fusion { name, .. } if name == DOT_ROW => {
                codegen::row_gemm_configs(step.output.1.shape.last().copied().unwrap_or(1))
            }
            _ if is_gemm(step) => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// A launch of `step` with `config`.
    fn build(&mut self, _: &Step, config: Config) -> Tuned {
        let run: Run =
            Box::new(move |step, inputs, out, outs| gemm(step, inputs, out, outs, Some(&config)));

        Tuned::new(format!("{config:?}"), run)
    }
}

/// Autotuning candidates for loop kernels, of the plan's `deterministic`.
pub(crate) struct Loops {
    pub(crate) deterministic: bool,
}

impl Kernels for Loops {
    type Config = LoopConfig;

    /// Loop configs whose block covers the kernel's lanes and splits; none for a scatter-add.
    fn configs(&self, step: &Step) -> Vec<LoopConfig> {
        let Some(kernel) = loop_kernel(step, self.deterministic) else {
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
    fn build(&mut self, _: &Step, config: LoopConfig) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Times steps on `device` with CUDA events, their loop kernels those of the plan's `deterministic`.
pub(crate) struct Events {
    pub(crate) device: Device,
    pub(crate) deterministic: bool,
}

impl Timer for Events {
    type Buffers = (Option<Kernel>, Vec<Tensor>, Tensor, Vec<Tensor>);

    /// The device steps run on.
    fn device(&self) -> Device {
        self.device
    }

    /// Zeroed inputs and outputs for `step`; scalars live on the CPU, and in-place ops write their operand.
    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String> {
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

        Ok(Some((
            loop_kernel(step, self.deterministic),
            inputs,
            out,
            outs,
        )))
    }

    /// One run of the step.
    fn run(
        &self,
        (kernel, inputs, out, outs): &Self::Buffers,
        step: &Step,
        _: usize,
    ) -> Result<(), String> {
        run(kernel.as_ref(), step, inputs, out, outs)
    }
}
