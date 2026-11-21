//! Running plan steps on TPU: each step's kernel (`codegen::kernel`), which codegen picked: its Pallas kernel,
//! generated when its plan compiles, lowered once per device and launched into the device's batch.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::{unsupported, viewed};
use crate::Device;
use crate::codegen::devices::tpu::Kernel;
use crate::codegen::kernel::Runner;
use crate::graph::Primitive;
use crate::plan::run::Buffers;
use crate::plan::{Buffer, Step, StepBase};

/// The raw buffers a TPU step runs on: its TPU, its inputs (then its further outputs) and its output.
pub(crate) struct Pointers {
    pub(crate) index: usize,
    pub(crate) inputs: Vec<*const u8>,
    pub(crate) output: *mut u8,
}

/// A step on a TPU: the step, the kernel it runs, and the Pallas kernel codegen generated for it.
#[derive(Debug, Clone)]
pub struct TpuStep {
    base: StepBase,
    /// The Pallas kernel codegen generated for the step; none when it has nothing to launch, or is a gather only
    /// the SparseCore may take.
    pub(crate) code: Option<Kernel>,
    /// The kernel the step runs.
    pub(crate) kernel: Arc<dyn Runner<TpuStep, Pointers>>,
}

impl TpuStep {
    /// `base`, running `kernel`, with the Pallas kernel `code` codegen generated for it.
    pub(crate) fn new(
        base: StepBase,
        code: Option<Kernel>,
        kernel: Arc<dyn Runner<TpuStep, Pointers>>,
    ) -> Self {
        TpuStep { base, code, kernel }
    }
}

impl Deref for TpuStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for TpuStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

impl Step for TpuStep {
    /// Runs the step's kernel on its TPU's buffers, launched into the device's batch.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let Device::Tpu(index) = buffers.device else {
            unreachable!("a TPU plan runs on a TPU")
        };

        let pointers = Pointers {
            index,
            inputs: buffers.args(self),
            output: buffers.pointer(&self.output),
        };

        self.kernel.run(self, self.tuned.as_ref(), &pointers)
    }

    /// Errors unless `device` is a TPU, no step reads runtime scalars or an operand through a view.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        let runs = |d| matches!(d, Device::Tpu(_));

        if let Some(e) = unsupported(steps, device, runs, "a TPU").or_else(|| viewed(steps)) {
            return Err(e);
        }

        let scalar = steps.iter().find(|s| {
            !matches!(s.primitive, Primitive::CustomCall { .. })
                && s.inputs.iter().any(|(b, _)| matches!(b, Buffer::Scalar(_)))
        });

        match scalar {
            Some(step) => Err(format!(
                "{}: runtime scalars do not run on TPU yet",
                step.label
            )),
            None => Ok(()),
        }
    }

    /// Runs `run` in a batch on the TPU, then the batch's launches as one program.
    fn batched(device: Device, run: &mut dyn FnMut() -> Result<(), String>) -> Result<(), String> {
        let Device::Tpu(index) = device else {
            unreachable!("a TPU plan runs on a TPU")
        };

        let batch = crate::runtime::stream::tpu::Batch::new(index);
        run()?;
        batch.run()
    }

    /// The step without its kernel.
    fn into_base(self) -> StepBase {
        self.base
    }
}
