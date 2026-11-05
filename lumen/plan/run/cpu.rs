//! Running plan steps on the CPU, through the reference kernels.

use std::ops::{Deref, DerefMut};

use super::Buffers;
use crate::Device;
use crate::graph::TensorType;
use crate::plan::step::{unsupported, viewed};
use crate::plan::{Step, StepBase};
use crate::reference;

/// A step the CPU runs with the reference kernels; every device's codegen lowers it to its own step.
#[derive(Debug, Clone)]
pub struct CpuStep(pub StepBase);

impl Deref for CpuStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.0
    }
}

impl DerefMut for CpuStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.0
    }
}

impl Step for CpuStep {
    /// Runs the step's reference kernel, or a host collective through the default group's store.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let args = buffers.args(self);
        let out = buffers.pointer(&self.output);
        let types: Vec<&TensorType> = self.inputs.iter().map(|(_, ty)| ty).collect();
        let _kernel = crate::profiler::record_host_kernel(self.label);
        crate::profiler::launched::launched(self.label, || None);

        if let Some(ranks) = crate::distributed::collective::ranks(&self.primitive) {
            let comm = crate::distributed::host::Host::new(ranks)?;
            let send = args.first().copied().unwrap_or(std::ptr::null());

            return unsafe {
                crate::distributed::collective::run(
                    &comm,
                    &self.primitive,
                    send,
                    out,
                    types.first().copied(),
                    &self.output.1,
                )
            };
        }

        unsafe { reference::eval_raw(&self.primitive, &args, &types, out, &self.output.1) };
        Ok(())
    }

    /// Errors unless `device` is the CPU and no step reads an operand through a view.
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        let runs = |d| matches!(d, Device::Cpu(_));

        match unsupported(steps, device, runs, "the CPU").or_else(|| viewed(steps)) {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step itself.
    fn into_base(self) -> StepBase {
        self.0
    }
}
