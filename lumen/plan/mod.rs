//! Lowers a graph to a `Plan`: an ordered list of kernel steps whose values are placed in
//! caller inputs, fresh outputs or one shared workspace, and runs it on a device.

mod compile;
mod run;
mod step;
#[cfg(test)]
pub(crate) mod tests;

pub use step::{Buffer, Memo, Step, Tuned};

use std::fmt;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

pub use crate::graph::View;
use crate::graph::{Primitive, TensorType};

/// Byte alignment of every value and scratch region in the workspace.
pub const ALIGNMENT: usize = 256;

/// Bytes of kernel scratch a primitive needs for the given operand and output types.
pub type ScratchFn = fn(&Primitive, &[&TensorType], &TensorType) -> usize;

/// Options controlling how `Plan::compile_with` places values.
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

/// A compiled program: its steps, where each value lives and the workspace size it needs.
#[derive(Debug, Clone)]
pub struct Plan {
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
    steps: Vec<Step>,
    workspace_bytes: usize,
    /// Alignment the workspace's base address needs.
    workspace_alignment: usize,
    /// Packed parameter blocks: member input positions and the dimension they are packed
    /// along; their block tensors follow the regular inputs.
    pub(crate) packed: Vec<(Vec<usize>, usize)>,
    /// Per packed block, the region sizes `Tensor::pack` interleaves its members with.
    pub(crate) interleaved: Vec<Vec<usize>>,
    /// Set when the program runs as host and device stages instead of `steps`.
    pub(crate) staged: Option<Box<Staged>>,
    /// Per step, the compiled Triton kernel used on CUDA.
    pub(crate) triton: Option<Vec<Option<crate::codegen::triton::Kernel>>>,
    /// Whether the CUDA (CuTe DSL) backend runs the steps, each a kernel generated at its launch.
    pub(crate) cuda: bool,
    /// MPS launches captured per step, replayed on later runs.
    #[cfg(lumen_mps_linked)]
    launches: Launches,
}

/// Per step, the prepared MPS launch once captured (`None` when it can't be replayed).
#[cfg(lumen_mps_linked)]
type Launches = OnceLock<Vec<OnceLock<Option<crate::runtime::mps::Prepared>>>>;

/// A program split into host and device stages, each with its own plan.
#[derive(Debug, Clone)]
pub(crate) struct Staged {
    /// The stages, in execution order.
    pub(crate) stages: Vec<Stage>,
    /// Where each program output comes from.
    pub(crate) outputs: Vec<Source>,
}

/// One host or device part of a staged program.
#[derive(Debug, Clone)]
pub(crate) struct Stage {
    /// Runs on the CPU rather than the device.
    pub(crate) host: bool,
    pub(crate) plan: Plan,
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

impl Plan {
    /// Types of all inputs, packed blocks included.
    pub(crate) fn input_types(&self) -> &[TensorType] {
        &self.inputs
    }

    /// The steps in execution order.
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Mutable steps; drops any captured MPS launches since they may no longer match.
    #[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
    pub(crate) fn steps_mut(&mut self) -> &mut [Step] {
        #[cfg(lumen_mps_linked)]
        {
            self.launches = OnceLock::new();
        }

        &mut self.steps
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

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(staged) = &self.staged {
            write!(f, "staged plan (workspace {} bytes)", self.workspace_bytes)?;

            for (s, stage) in staged.stages.iter().enumerate() {
                let side = if stage.host { "host" } else { "device" };

                write!(
                    f,
                    "\nstage {s} ({side}, inputs {:?}): {}",
                    stage.inputs, stage.plan
                )?;
            }

            return write!(f, "\noutputs {:?}", staged.outputs);
        }

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
