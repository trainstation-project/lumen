//! Running CUDA (CuTe DSL) steps: each step's kernel (`codegen::kernel`), which codegen picked and compiles for its
//! tensors' layouts at its first launch on them.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::triton::traced;
use super::{Tensors, unsupported};
use crate::autotune::{KernelStep, Timer};
use crate::codegen::kernel::Runner;
use crate::plan::run::Buffers;
use crate::plan::{Step, StepBase};
use crate::{Device, Tensor, TensorOptions};

/// A CUDA (CuTe DSL) step: the step and the kernel it runs.
#[derive(Debug, Clone)]
pub struct CudaStep {
    /// The step's primitive and buffers.
    pub base: StepBase,
    /// The kernel the step runs: the GEMM, the attention forward, or a generated elementwise kernel.
    pub(crate) kernel: Arc<dyn Runner<CudaStep, Tensors>>,
}

impl Deref for CudaStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for CudaStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

impl Step for CudaStep {
    /// Runs the step's kernel on its tensors under its tuned config.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let tensors = buffers.tensors(self);
        traced(self, &tensors.1, || {
            self.kernel.run(self, self.tuned.as_ref(), &tensors)
        })
    }

    /// Errors unless `device` is a CUDA device.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        let runs = |d| matches!(d, Device::Cuda(_));

        match unsupported(steps, device, runs, "CUDA") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step itself.
    fn into_base(self) -> StepBase {
        self.base
    }
}

/// A step tunes its kernel's configs.
impl KernelStep for CudaStep {
    type Buffers = Tensors;

    /// The step's kernel.
    fn kernel(&self) -> &dyn Runner<CudaStep, Tensors> {
        self.kernel.as_ref()
    }
}

/// Times steps on `device` with CUDA events.
pub(crate) struct CudaTimer {
    pub(crate) device: Device,
}

impl Timer for CudaTimer {
    type Step = CudaStep;
    type Buffers = Tensors;

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

    /// One run of the step's kernel under its trial config.
    fn run(&self, tensors: &Tensors, step: &CudaStep, _: usize) -> Result<(), String> {
        step.kernel.run(step, step.tuned.as_ref(), tensors)
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
