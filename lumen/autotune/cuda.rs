//! CUDA (CuTe DSL) autotuning: the Ampere GEMM's tile candidates and the attention forward's configs, timed with
//! CUDA events.

use super::{Kernels, Timer};
use crate::codegen::attention::cuda as attention;
use crate::codegen::dot_general::cuda::{CONFIGS, Config, TUNED_MACS};
use crate::graph::Primitive;
use crate::graph_compiler::passes::attention::{FORWARD, of_body};
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

    /// Whether `step` is a matmul (the GEMM runs every one, tuned only from `TUNED_MACS` up).
    fn handles(&self, step: &CudaStep) -> bool {
        matches!(step.primitive, Primitive::DotGeneral { .. })
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &CudaStep, config: Config) -> Tuned {
        Tuned::new(format!("{config:?}"), config)
    }
}

/// Autotuning candidates for flash-attention forward steps.
pub(crate) struct AttentionForwards;

impl Kernels for AttentionForwards {
    type Step = CudaStep;
    type Config = attention::Config;

    /// Every config that fits the head dims of a forward attention step of at least `attention::TUNED_MACS`
    /// multiply-adds, none otherwise.
    fn configs(&self, step: &CudaStep) -> Vec<attention::Config> {
        let Primitive::Fusion { name, body, .. } = &step.primitive else {
            return Vec::new();
        };

        match of_body(body).filter(|_| name == FORWARD) {
            Some(a) => {
                let macs = a.batch.iter().product::<usize>() * a.sq * a.sk * (a.h + a.hv);

                match macs >= attention::TUNED_MACS {
                    true => attention::configs(a.h, a.hv),
                    false => Vec::new(),
                }
            }
            None => Vec::new(),
        }
    }

    /// Whether `step` is a forward attention (the kernel runs every one, tuned only from `TUNED_MACS` up).
    fn handles(&self, step: &CudaStep) -> bool {
        matches!(&step.primitive, Primitive::Fusion { name, .. } if name == FORWARD)
    }

    /// Whether forced `config` fits `step`'s head dims.
    fn accepts(&self, step: &CudaStep, config: &attention::Config) -> bool {
        let Primitive::Fusion { body, .. } = &step.primitive else {
            return false;
        };

        self.handles(step) && of_body(body).is_some_and(|a| attention::fits(config, a.h, a.hv))
    }

    /// The step launched with `config`.
    fn build(&mut self, _: &CudaStep, config: attention::Config) -> Tuned {
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

    /// Inputs and outputs for `step`, all on the device (the CUDA backend passes runtime scalars as 0-d device
    /// tensors): float inputs hold values in [-1, 1), since a GPU's clocks under math on zeros differ from those
    /// under real data; every other buffer is zeroed (integer inputs may be indices, which must stay in bounds).
    fn buffers(&self, step: &CudaStep) -> Result<Option<Self::Buffers>, String> {
        let options = |ty: &crate::graph::TensorType| {
            TensorOptions::new().dtype(ty.dtype).device(self.device)
        };
        let zeros = |ty: &crate::graph::TensorType| Tensor::zeros(&ty.shape, options(ty));

        let input = |ty: &crate::graph::TensorType| match ty.dtype.is_float() {
            true => Tensor::from_slice(&noise(ty.shape.iter().product()), options(ty))
                .reshape(&ty.shape),
            false => zeros(ty),
        };

        let inputs = step.inputs.iter().map(|(_, ty)| input(ty)).collect();
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

/// `n` pseudo-random values in [-1, 1), the same every call (a 64-bit LCG's high bits).
fn noise(n: usize) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);

            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}
