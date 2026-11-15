//! Running plan steps on the CPU: each step's kernel (`codegen::kernel`), its reference implementation or a host
//! collective.

use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::{unsupported, viewed};
use crate::Device;
use crate::codegen::kernel::Runner;
use crate::graph::TensorType;
use crate::plan::run::Buffers;
use crate::plan::{Step, StepBase};
use crate::reference;

/// The raw buffers a CPU step runs on: its inputs and its output.
pub(crate) struct Pointers {
    pub(crate) inputs: Vec<*const u8>,
    pub(crate) output: *mut u8,
}

/// A step the CPU runs: the step and the kernel it runs; every device's codegen lowers it to its own step.
#[derive(Debug, Clone)]
pub struct CpuStep {
    /// The step's primitive and buffers.
    pub base: StepBase,
    /// The kernel the step runs: a host collective, else the reference implementation.
    pub(crate) kernel: Arc<dyn Runner<CpuStep, Pointers>>,
}

impl CpuStep {
    /// `base`, running a host collective for a collective's primitive, else its reference implementation.
    pub(crate) fn new(base: StepBase) -> Self {
        let kernel: Arc<dyn Runner<CpuStep, Pointers>> =
            match crate::distributed::collective::ranks(&base.primitive) {
                Some(_) => Arc::new(HostCollective),
                None => Arc::new(Reference),
            };

        CpuStep { base, kernel }
    }
}

impl Deref for CpuStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for CpuStep {
    /// The step's primitive and buffers.
    fn deref_mut(&mut self) -> &mut StepBase {
        &mut self.base
    }
}

impl Step for CpuStep {
    /// Runs the step's kernel on its buffers, recorded as a host kernel.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let pointers = Pointers {
            inputs: buffers.args(self),
            output: buffers.pointer(&self.output),
        };

        let _kernel = crate::profiler::record_host_kernel(self.label);
        crate::profiler::launched::launched(self.label, || None);
        self.kernel.run(self, self.tuned.as_ref(), &pointers)
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
        self.base
    }
}

/// A host collective (an all-reduce and the like across processes), through the default group's store.
pub(crate) struct HostCollective;

impl crate::codegen::kernel::Kernel for HostCollective {
    type Step = CpuStep;
    type Buffers = Pointers;
    type Config = ();

    /// Runs the collective over its ranks, sending its operand (if any) and receiving into the output.
    fn run(&self, step: &CpuStep, _: Option<&()>, p: &Pointers) -> Result<(), String> {
        let ranks = crate::distributed::collective::ranks(&step.primitive).expect("a collective");
        let comm = crate::distributed::host::Host::new(ranks)?;
        let send = p.inputs.first().copied().unwrap_or(std::ptr::null());
        let operand = step.inputs.first().map(|(_, ty)| ty);

        unsafe {
            crate::distributed::collective::run(
                &comm,
                &step.primitive,
                send,
                p.output,
                operand,
                &step.output.1,
            )
        }
    }
}

/// The reference implementation of a step's primitive (`reference::eval_raw`).
pub(crate) struct Reference;

impl crate::codegen::kernel::Kernel for Reference {
    type Step = CpuStep;
    type Buffers = Pointers;
    type Config = ();

    /// Evaluates the primitive on its operands into the output.
    fn run(&self, step: &CpuStep, _: Option<&()>, p: &Pointers) -> Result<(), String> {
        let types: Vec<&TensorType> = step.inputs.iter().map(|(_, ty)| ty).collect();
        unsafe {
            reference::eval_raw(&step.primitive, &p.inputs, &types, p.output, &step.output.1)
        };
        Ok(())
    }
}
