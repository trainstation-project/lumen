//! Lowers a graph to a plan: an ordered list of kernel steps whose values are placed in
//! caller inputs, fresh outputs or one shared workspace, and runs it on its device. Each device has its own
//! step type (`run::<device>`) and plan (`DevicePlan` of its steps); a staged plan runs host and device plans.

mod compile;
pub(crate) mod run;
mod step;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(lumen_mps_linked)]
pub(crate) use run::mps::MpsStep;
#[cfg(lumen_tpu_linked)]
pub(crate) use run::tpu::TpuStep;
pub(crate) use run::{cpu::CpuStep, cuda::CudaStep, triton::TritonStep};
pub use step::{Buffer, Memo, Step, StepBase, Tuned};

use std::fmt;
use std::ops::{Deref, DerefMut};

pub use crate::graph::View;
use crate::graph::{Primitive, TensorType};
use crate::{Device, Tensor};

/// Byte alignment of every value and scratch region in the workspace.
pub const ALIGNMENT: usize = 256;

/// Bytes of kernel scratch a primitive needs for the given operand and output types.
pub type ScratchFn = fn(&Primitive, &[&TensorType], &TensorType) -> usize;

/// Options controlling how `CpuPlan::compile_with` places values.
#[derive(Debug, Clone, Default)]
pub struct PlanOptions {
    /// Sizes the scratch each step needs; it is reserved in the workspace.
    pub scratch: Option<ScratchFn>,
    /// Inputs whose buffers may hold any output of the same type.
    pub donate: Vec<usize>,
    /// `(input, output)` pairs: the input's buffer may hold only that output.
    pub donate_into: Vec<(usize, usize)>,
    /// Which inputs are parameters; `Some` makes an owned plan whose outputs stay in the
    /// workspace or inputs and that runs with `Plan::run_in`.
    pub parameters: Option<Vec<bool>>,
    /// Inputs passed as host scalars instead of device buffers.
    pub scalars: Vec<usize>,
    /// Workspace bytes above which values are rematerialized instead of kept.
    pub memory_limit: Option<usize>,
    /// `(block input, member inputs)` of packed parameter blocks; a member is not
    /// overwritten while its block is still read.
    pub blocks: Vec<(usize, Vec<usize>)>,
    /// Device-specific storage sizes and alignment; `None` means dense row-major.
    pub layout: Option<DeviceLayout>,
}

/// How a device stores tensors, for backends whose layout isn't dense row-major.
#[derive(Debug, Clone, Copy)]
pub struct DeviceLayout {
    /// Bytes a value of the type occupies.
    pub bytes: fn(&TensorType) -> usize,
    /// Whether two types share a storage layout, so a reshape between them moves no data.
    pub same: fn(&TensorType, &TensorType) -> bool,
    /// Alignment a value of the type needs.
    pub align: fn(&TensorType) -> usize,
}

/// Least common multiple of `a` and `b`.
pub(crate) fn lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a, b);

    while y != 0 {
        (x, y) = (y, x % y);
    }

    a / x * b
}

/// What every plan shares: where each value lives and the workspace size it needs.
#[derive(Debug, Clone)]
pub struct PlanBase {
    /// Types of the inputs, including packed blocks after the program's own.
    inputs: Vec<TensorType>,
    outputs: Vec<TensorType>,
    /// Per output, the input whose buffer it is written into.
    aliases: Vec<Option<usize>>,
    /// Where each input is read from.
    inputs_at: Vec<Buffer>,
    /// Per input, whether an owned plan treats it as a caller argument rather than a
    /// parameter, copying it when needed.
    arguments: Vec<bool>,
    /// Where each output is written; `Output(k)` unless the plan is owned.
    outputs_at: Vec<Buffer>,
    workspace_bytes: usize,
    /// Alignment the workspace's base address needs.
    workspace_alignment: usize,
    /// Packed parameter blocks: member input positions and the dimension they are packed
    /// along; their block tensors follow the regular inputs.
    pub(crate) packed: Vec<(Vec<usize>, usize)>,
    /// Per packed block, the region sizes `Tensor::pack` interleaves its members with.
    pub(crate) interleaved: Vec<Vec<usize>>,
}

/// A compiled program on its device: the `PlanBase` it derefs to, its steps, and how it runs.
pub trait Plan:
    Deref<Target = PlanBase> + DerefMut + fmt::Display + fmt::Debug + Send + Sync
{
    /// The steps in execution order (a staged plan's, every stage's in turn).
    fn steps(&self) -> Vec<&dyn Step>;

    /// The host and device stages, when the program runs as stages instead of steps.
    fn staged(&self) -> Option<&Staged> {
        None
    }

    /// Runs on `device`, allocating the workspace and outputs; all inputs must already be there.
    /// On the meta device it only returns empty outputs.
    fn run_on(&self, inputs: &[Tensor], device: Device) -> Result<Vec<Tensor>, String>;

    /// Runs an owned plan in a caller-provided workspace; outputs placed in the workspace are
    /// returned as views of it, so the next run overwrites them.
    fn run_in(&self, workspace: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String>;

    /// Runs on the device of the first input (the CPU when there are none).
    fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.run_on(
            inputs,
            inputs.first().map_or(Device::Cpu(0), Tensor::device),
        )
    }
}

/// A plan of one device's steps.
#[derive(Debug, Clone)]
pub struct DevicePlan<S> {
    base: PlanBase,
    steps: Vec<S>,
}

/// A plan the CPU runs with the reference kernels; a device's codegen lowers it to its own plan.
pub type CpuPlan = DevicePlan<CpuStep>;
/// A plan of Metal kernels.
#[cfg(lumen_mps_linked)]
pub type MpsPlan = DevicePlan<MpsStep>;
/// A plan of Pallas kernels on a TPU.
#[cfg(lumen_tpu_linked)]
pub type TpuPlan = DevicePlan<TpuStep>;
/// A plan of Triton kernels on CUDA.
pub type TritonPlan = DevicePlan<TritonStep>;
/// A plan of CUDA (CuTe DSL) kernels.
pub type CudaPlan = DevicePlan<CudaStep>;

impl<S> Deref for DevicePlan<S> {
    type Target = PlanBase;

    /// The plan's buffers.
    fn deref(&self) -> &PlanBase {
        &self.base
    }
}

impl<S> DerefMut for DevicePlan<S> {
    /// The plan's buffers.
    fn deref_mut(&mut self) -> &mut PlanBase {
        &mut self.base
    }
}

impl<S: Step> DevicePlan<S> {
    /// The steps in execution order.
    pub fn steps(&self) -> &[S] {
        &self.steps
    }

    /// Mutable steps.
    pub(crate) fn steps_mut(&mut self) -> &mut [S] {
        &mut self.steps
    }

    /// The plan with each step made into another device's by `f`, in order.
    pub(crate) fn map<T>(
        self,
        f: impl FnMut(S) -> Result<T, String>,
    ) -> Result<DevicePlan<T>, String> {
        Ok(DevicePlan {
            base: self.base,
            steps: self.steps.into_iter().map(f).collect::<Result<_, _>>()?,
        })
    }
}

/// A program split into host and device stages, each with its own plan.
#[derive(Debug)]
pub struct Staged {
    /// The stages, in execution order.
    pub(crate) stages: Vec<Stage>,
    /// Where each program output comes from.
    pub(crate) outputs: Vec<Source>,
}

/// A plan of host and device stages.
#[derive(Debug)]
pub struct StagedPlan {
    base: PlanBase,
    staged: Staged,
}

impl Deref for StagedPlan {
    type Target = PlanBase;

    /// The program's inputs, outputs and shared workspace.
    fn deref(&self) -> &PlanBase {
        &self.base
    }
}

impl DerefMut for StagedPlan {
    /// The program's inputs, outputs and shared workspace.
    fn deref_mut(&mut self) -> &mut PlanBase {
        &mut self.base
    }
}

/// One host or device part of a staged program.
#[derive(Debug)]
pub(crate) struct Stage {
    /// Runs on the CPU rather than the device.
    pub(crate) host: bool,
    pub(crate) plan: Box<dyn Plan>,
    /// Where each of the stage's inputs comes from.
    pub(crate) inputs: Vec<Source>,
    /// Indices of the parent plan's packed blocks this stage reads, after `inputs`.
    pub(crate) blocks: Vec<usize>,
    /// Byte offset of this stage's workspace within the parent workspace.
    pub(crate) offset: usize,
}

/// Where a staged value comes from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Source {
    /// Program input `i`.
    Input(usize),
    /// Output `k` of stage `s`.
    Stage(usize, usize),
    /// Output `k` of stage `s`, copied to the CPU.
    ToHost(usize, usize),
    /// Output `k` of stage `s`, copied to the run's device.
    ToDevice(usize, usize),
}

impl PlanBase {
    /// Types of all inputs, packed blocks included.
    pub(crate) fn input_types(&self) -> &[TensorType] {
        &self.inputs
    }

    /// Workspace bytes a run needs.
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Alignment the workspace must have.
    pub fn workspace_alignment(&self) -> usize {
        self.workspace_alignment
    }

    /// Packed parameter blocks as `(member input positions, dimension)`.
    pub fn packed(&self) -> &[(Vec<usize>, usize)] {
        &self.packed
    }

    /// Per packed block, the region sizes its members are interleaved with.
    pub fn interleaved(&self) -> &[Vec<usize>] {
        &self.interleaved
    }
}

impl<S: Step> fmt::Display for DevicePlan<S> {
    /// The plan's listing: its workspace, then each step's outputs, primitive, operands and scratch.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "plan (workspace {} bytes)", self.workspace_bytes)?;

        for step in &self.steps {
            let (out, ty) = &step.output;
            write!(f, "\n    {out}:{ty}")?;

            for (b, ty) in &step.extra_outputs {
                write!(f, ", {b}:{ty}")?;
            }

            write!(f, " = {}", step.primitive)?;

            for ((b, _), v) in step.inputs.iter().zip(&step.views) {
                write!(f, " {b}")?;

                if let Some(View { offset, strides }) = v {
                    write!(f, "[+{offset} strides {strides:?}]")?;
                }
            }

            if let Some((offset, bytes)) = step.scratch {
                write!(f, " (scratch ws+{offset}, {bytes} bytes)")?;
            }
        }

        Ok(())
    }
}

impl fmt::Display for StagedPlan {
    /// The plan's listing: its workspace, each stage's plan, and where the outputs come from.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "staged plan (workspace {} bytes)", self.workspace_bytes)?;

        for (s, stage) in self.staged.stages.iter().enumerate() {
            let side = if stage.host { "host" } else { "device" };

            write!(
                f,
                "\nstage {s} ({side}, inputs {:?}): {}",
                stage.inputs, stage.plan
            )?;
        }

        write!(f, "\noutputs {:?}", self.staged.outputs)
    }
}
