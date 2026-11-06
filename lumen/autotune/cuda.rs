//! CUDA (CuTe DSL) autotuning: the Ampere GEMM's tile candidates, timed with CUDA events.

use super::{Kernels, Timer};
use crate::codegen::dot_general::cuda::{CONFIGS, Config, TUNED_MACS};
use crate::graph::Primitive;
use crate::plan::{CudaStep, Tuned};
use crate::{Device, Tensor, TensorOptions};

/// Autotuning candidates for matmul steps.
pub(crate) struct Gemms;

impl Kernels for Gemms {
    type Step = CudaStep;
    type Config = Config;

    /// Every GEMM tile for a matmul step of at least `TUNED_MACS` multiply-adds, none otherwise.
    fn configs(&self, step: &CudaStep) -> Vec<Config> {
        let Primitive::DotGeneral {
            lhs_contracting, ..
        } = &step.primitive
        else {
            return Vec::new();
        };

        let k = step.inputs[0].1.shape[lhs_contracting[0]];
        let macs = step.output.1.shape.iter().product::<usize>() * k;

        match macs >= TUNED_MACS {
            true => CONFIGS.to_vec(),
            false => Vec::new(),
        }
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &CudaStep, config: Config) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Times steps on `device` with CUDA events.
pub(crate) struct Events {
    pub(crate) device: Device,
}

impl Timer for Events {
    type Step = CudaStep;
    type Buffers = (Vec<Tensor>, Tensor, Vec<Tensor>);

    /// The device steps run on.
    fn device(&self) -> Device {
        self.device
    }

    /// Zeroed inputs and outputs for `step`, all on the device (the CUDA backend passes runtime scalars as 0-d
    /// device tensors).
    fn buffers(&self, step: &CudaStep) -> Result<Option<Self::Buffers>, String> {
        let zeros = |ty: &crate::graph::TensorType| {
            Tensor::zeros(
                &ty.shape,
                TensorOptions::new().dtype(ty.dtype).device(self.device),
            )
        };

        let inputs = step.inputs.iter().map(|(_, ty)| zeros(ty)).collect();
        let outs = step.extra_outputs.iter().map(|(_, ty)| zeros(ty)).collect();
        Ok(Some((inputs, zeros(&step.output.1), outs)))
    }

    /// One run of the step.
    fn run(
        &self,
        (inputs, out, outs): &Self::Buffers,
        step: &CudaStep,
        _: usize,
    ) -> Result<(), String> {
        step.run_raw(inputs, out, outs)
    }
}
