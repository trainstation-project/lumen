//! Running plan steps on CUDA through Triton: each step's kernel (`codegen::kernel`), which codegen picked, and the
//! runtime kernels (collectives and waits) beside the generated ones.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::{Tensors, unsupported};
use crate::autotune::{KernelStep, Timer};
use crate::codegen::devices::triton::Kernel;
use crate::codegen::elementwise::triton::copy;
use crate::codegen::kernel::Runner;
use crate::graph::{Primitive, TensorType};
use crate::plan::run::Buffers;
use crate::plan::{Buffer, Step, StepBase};
use crate::{Device, Tensor, TensorOptions};

/// A step of Triton kernels on CUDA: the step, the kernel it runs, and the loop kernel codegen generated for it.
#[derive(Debug, Clone)]
pub struct TritonStep {
    base: StepBase,
    /// The loop kernel codegen generated for the step; none for a step another kernel (a GEMM's, flash
    /// attention's, a collective) runs.
    pub(crate) code: Option<Kernel>,
    /// The kernel the step runs.
    pub(crate) kernel: Arc<dyn Runner<TritonStep, Tensors>>,
}

impl TritonStep {
    /// `base`, running `kernel`, with the loop kernel `code` codegen generated for it.
    pub(crate) fn new(
        base: StepBase,
        code: Option<Kernel>,
        kernel: Arc<dyn Runner<TritonStep, Tensors>>,
    ) -> Self {
        TritonStep { base, code, kernel }
    }
}

impl Deref for TritonStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for TritonStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

/// Runs `run`, `step`'s launch writing `out`: captured into a CUDA graph, the nodes it adds are the step's, by
/// name and types; profiled, its kernels are timed under its name (each keeps its own as `kernel`) with its types.
pub(crate) fn traced(
    step: &StepBase,
    out: &Tensor,
    run: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    #[cfg(all(feature = "python", lumen_cuda_linked))]
    let run = || crate::runtime::kernel::cuda::driver::counted(step, run);

    #[cfg(lumen_cupti_linked)]
    let run = || {
        crate::profiler::next_launch(step.label, || step.io_types());
        crate::profiler::cupti::correlated(out.device(), run)
    };

    let _ = (step, out);
    run()
}

impl Step for TritonStep {
    /// Runs the step's kernel on its tensors under its tuned config.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let tensors = buffers.tensors(self);
        traced(self, &tensors.1, || {
            self.kernel.run(self, self.tuned.as_ref(), &tensors)
        })
    }

    /// Whether the step's kernel reads operand `k`, a caller input of `shape`, as laid out (attention any layout
    /// with a contiguous last dim, copying others itself).
    fn reads_strided(&self, k: usize, shape: &[usize], _: &[usize]) -> bool {
        self.inputs[k].1.shape == shape
            && matches!(
                self.primitive,
                Primitive::DotGeneral { .. }
                    | Primitive::Fusion { .. }
                    | Primitive::Cumsum { .. }
                    | Primitive::LinearCrossEntropy
            )
    }

    /// Errors unless `device` is a CUDA device.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        match unsupported(steps, device, |d| matches!(d, Device::Cuda(_)), "CUDA") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step without its kernel.
    fn into_base(self) -> StepBase {
        self.base
    }
}

/// A step tunes its kernel's configs.
impl KernelStep for TritonStep {
    type Buffers = Tensors;

    /// The step's kernel.
    fn kernel(&self) -> &dyn Runner<TritonStep, Tensors> {
        self.kernel.as_ref()
    }
}

/// A collective (an all-reduce and the like), run by NCCL.
pub(crate) struct Collective;

impl crate::codegen::kernel::Kernel for Collective {
    type Step = TritonStep;
    type Buffers = Tensors;
    type Config = ();

    /// Runs the collective on its operand into `out`.
    fn run(
        &self,
        step: &TritonStep,
        _: Option<&()>,
        (inputs, out, _): &Tensors,
    ) -> Result<(), String> {
        #[cfg(lumen_cuda_linked)]
        return crate::distributed::nccl::run(&step.primitive, inputs, out);

        #[cfg(not(lumen_cuda_linked))]
        {
            let _ = (inputs, out);
            Err(format!("{}: has no Triton kernel on CUDA yet", step.label))
        }
    }
}

/// A wait for the device's collectives (`DeviceWait`) or for the host (`HostWait`) before its operand is read,
/// passing the operand through.
pub(crate) struct Wait;

impl crate::codegen::kernel::Kernel for Wait {
    type Step = TritonStep;
    type Buffers = Tensors;
    type Config = ();

    /// Waits as the step's primitive says, then writes its operand to `out` unless it is already there.
    fn run(
        &self,
        step: &TritonStep,
        _: Option<&()>,
        (inputs, out, _): &Tensors,
    ) -> Result<(), String> {
        match step.primitive {
            Primitive::DeviceWait => {
                #[cfg(lumen_cuda_linked)]
                crate::distributed::nccl::device_wait(&inputs[0])?;
            }
            _ => crate::runtime::stream::stream_for(out.device())?.synchronize(),
        }

        match out.data_ptr() == inputs[0].data_ptr() {
            true => Ok(()),
            false => copy(out, &inputs[0]),
        }
    }
}

/// A step no Triton kernel runs yet: it fails when it runs.
pub(crate) struct Unsupported;

impl crate::codegen::kernel::Kernel for Unsupported {
    type Step = TritonStep;
    type Buffers = Tensors;
    type Config = ();

    /// Fails: the step has no Triton kernel.
    fn run(&self, step: &TritonStep, _: Option<&()>, _: &Tensors) -> Result<(), String> {
        Err(format!("{}: has no Triton kernel on CUDA yet", step.label))
    }
}

/// A hash of `tensors`' dtypes, shapes and strides: the key of what a step generated for those layouts.
pub(crate) fn layouts<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    for t in tensors {
        (t.dtype(), t.shape(), t.strides()).hash(&mut hasher);
    }

    hasher.finish()
}

/// Times steps on `device` with CUDA events.
pub(crate) struct TritonTimer {
    pub(crate) device: Device,
}

impl Timer for TritonTimer {
    type Step = TritonStep;
    type Buffers = Tensors;

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

    /// One run of the step's kernel under its trial config.
    fn run(&self, tensors: &Tensors, step: &TritonStep, _: usize) -> Result<(), String> {
        step.kernel.run(step, step.tuned.as_ref(), tensors)
    }
}
