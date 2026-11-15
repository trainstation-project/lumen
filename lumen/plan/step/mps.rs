//! Running plan steps on MPS: each step's kernel (`codegen::kernel`), which codegen picked, encoded into the MPS
//! stream, its launches captured on its first run and replayed with new pointers on later ones; and the runtime
//! kernels (host waits) beside the generated ones.

use super::unsupported;
use crate::allocator::Scratch;
use crate::autotune::{KernelStep, Timer};
use crate::codegen::elementwise::mps::Tunable;
use crate::codegen::kernel::Runner;
use crate::graph::{NEURAL_ENGINE, Primitive, TensorType, View};
use crate::plan::run::Buffers;
use crate::plan::{Buffer, Step, StepBase};
use crate::runtime::kernel::mps::{Pointers, Prepared, Unprepared, capture, prepare, replay};
use crate::runtime::wait::wait;
use crate::{Device, Tensor};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, OnceLock};

/// The raw buffers an MPS step's kernels are encoded on: its inputs (then its further outputs), its output and its
/// scratch, and the tensors its launches keep alive.
pub(crate) struct Encoding {
    pub(crate) inputs: Vec<*const u8>,
    pub(crate) output: *mut u8,
    pub(crate) scratch: *mut u8,
    pub(crate) keep: Vec<Tensor>,
}

/// A step of Metal kernels: the kernel it runs, what an elementwise fusion needs to regenerate its kernel, and its
/// launches once captured.
#[derive(Debug, Clone)]
pub struct MpsStep {
    base: StepBase,
    /// What an elementwise fusion needs to regenerate its kernel under another tuning.
    pub(crate) code: Option<Tunable>,
    /// The kernel the step runs.
    pub(crate) kernel: Arc<dyn Runner<MpsStep, Encoding>>,
    /// The step's launches, prepared for replay once captured (`None` when they can't be replayed); any change
    /// to the step drops them.
    captured: OnceLock<Option<Prepared>>,
}

impl MpsStep {
    /// `base`, running `kernel`, with what codegen kept to regenerate its elementwise kernel (`code`).
    pub(crate) fn new(
        base: StepBase,
        code: Option<Tunable>,
        kernel: Arc<dyn Runner<MpsStep, Encoding>>,
    ) -> Self {
        MpsStep {
            base,
            code,
            kernel,
            captured: OnceLock::new(),
        }
    }
}

impl Deref for MpsStep {
    type Target = StepBase;

    /// The step's primitive and buffers.
    fn deref(&self) -> &StepBase {
        &self.base
    }
}

impl DerefMut for MpsStep {
    /// The step's primitive and buffers; its captured launches are dropped, as they may no longer match.
    fn deref_mut(&mut self) -> &mut StepBase {
        self.captured = OnceLock::new();
        &mut self.base
    }
}

/// Whether an MPS step's launch can be captured once and replayed with new pointers:
/// host-side work, custom calls and runtime scalar reads cannot.
fn replayable(step: &StepBase) -> bool {
    use Primitive::*;

    let launches = match &step.primitive {
        Fusion { name, .. } => !name.starts_with(NEURAL_ENGINE),
        HostWait | DeviceWait | ToHost | ToDevice | CustomCall { .. } | PallasCall { .. } => false,
        _ => true,
    };

    launches
        && !step
            .inputs
            .iter()
            .any(|(b, _)| matches!(b, Buffer::Scalar(_)))
}

/// Bytes from the start of a (possibly strided) operand to one past its last element.
fn extent(ty: &TensorType, view: Option<&View>) -> usize {
    let size = ty.dtype.size_of();

    match view {
        None => ty.numel() * size,
        Some(view) => {
            let last: usize = ty
                .shape
                .iter()
                .zip(&view.strides)
                .map(|(&n, &stride)| n.saturating_sub(1) * stride)
                .sum();

            (last + 1) * size
        }
    }
}

impl Step for MpsStep {
    /// Encodes the step's kernels, capturing its launches on its first run and replaying them with new pointers
    /// on later ones.
    fn run(&self, buffers: &Buffers) -> Result<(), String> {
        let step = &self.base;
        let args = buffers.args(step);
        let out = buffers.pointer(&step.output);
        let scratch = buffers.scratch(step);

        let mut keep: Vec<Tensor> = step
            .inputs
            .iter()
            .chain([&step.output])
            .chain(&step.extra_outputs)
            .map(|&(b, _)| buffers.tensor(b).holding())
            .collect();

        if step.scratch.is_some() {
            keep.push(buffers.workspace.holding());
        }

        // Capture the step's launch once and replay it with new pointers on later runs.
        if buffers.viewed || crate::profiler::is_enabled() || !replayable(step) {
            return self.run_raw(&args, out, scratch, keep);
        }

        let sizes = step
            .inputs
            .iter()
            .zip(&step.views)
            .map(|(s, v)| extent(&s.1, v.as_ref()))
            .chain(step.extra_outputs.iter().map(|s| extent(&s.1, None)));

        let operands: Vec<(*const u8, usize)> = args.iter().copied().zip(sizes).collect();
        let workspace = buffers.workspace;

        let pointers = Pointers {
            workspace: (workspace.data_ptr().cast_const(), workspace.nbytes()),
            operands: &operands,
            output: (out.cast_const(), extent(&step.output.1, None)),
        };

        match self.captured.get() {
            Some(Some(prepared)) => replay(prepared, &pointers, keep),
            Some(None) => self.run_raw(&args, out, scratch, keep),
            None => {
                let captured = capture(|| self.run_raw(&args, out, scratch, Vec::new()))?;

                match prepare(captured, &pointers) {
                    Ok(prepared) => {
                        replay(&prepared, &pointers, keep)?;
                        let _ = self.captured.set(Some(prepared));
                        Ok(())
                    }
                    Err(unprepared) => {
                        if let Unprepared::Unplaced = unprepared {
                            let _ = self.captured.set(None);
                        }

                        self.run_raw(&args, out, scratch, keep)
                    }
                }
            }
        }
    }

    /// Whether the step's kernel reads operand `k`, a caller input of `shape`, at `strides`.
    fn reads_strided(&self, k: usize, shape: &[usize], strides: &[usize]) -> bool {
        let view = &self.views[k];

        match &self.primitive {
            Primitive::DotGeneral { .. } => {
                let operands = [&self.inputs[0].1, &self.inputs[1].1];

                view.is_none()
                    && operands[k].shape == shape
                    && crate::codegen::dot_general::mps::reads_strided(
                        &self.primitive,
                        operands,
                        k,
                        strides,
                    )
            }
            Primitive::Fusion { name, .. } if name.starts_with(NEURAL_ENGINE) => true,
            Primitive::Fusion { body, .. } => {
                view.is_none()
                    && self.inputs[k].1.shape == shape
                    && (crate::codegen::mps::fusion_reads_strided(body, k, strides)
                        || crate::codegen::mps::loads_strided(body, k)
                        || crate::codegen::mps::attention_reads_strided(body, strides))
            }
            _ => false,
        }
    }

    /// Errors unless `device` is MPS (compiling for it rejected the values Metal has no type for).
    fn check(steps: &[Self], device: Device) -> Result<(), String> {
        match unsupported(steps, device, |d| d == Device::Mps(0), "MPS") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// The step without its kernel and captured launches.
    fn into_base(self) -> StepBase {
        self.base
    }
}

/// A step tunes its kernel's configs.
impl KernelStep for MpsStep {
    type Buffers = Encoding;

    /// The step's kernel.
    fn kernel(&self) -> &dyn Runner<MpsStep, Encoding> {
        self.kernel.as_ref()
    }
}

/// A host wait: blocks until the first kept tensor's device is done.
pub(crate) struct HostWait;

impl crate::codegen::kernel::Kernel for HostWait {
    type Step = MpsStep;
    type Buffers = Encoding;
    type Config = ();

    /// Waits for the device the first kept tensor is on.
    fn run(&self, _: &MpsStep, _: Option<&()>, e: &Encoding) -> Result<(), String> {
        wait(&e.keep[0]);
        Ok(())
    }
}

/// A step no Metal kernel runs: a concatenation (which runs fused), a collective (MPS meshes are one process), or a
/// primitive the plan never makes a step of on MPS.
pub(crate) struct Unsupported;

impl crate::codegen::kernel::Kernel for Unsupported {
    type Step = MpsStep;
    type Buffers = Encoding;
    type Config = ();

    /// Fails with why the step has no Metal kernel.
    fn run(&self, step: &MpsStep, _: Option<&()>, _: &Encoding) -> Result<(), String> {
        Err(match &step.primitive {
            Primitive::Concatenate { .. } => {
                "concatenate runs in a fusion on MPS: compile with fuse".into()
            }
            p if p.is_collective() || matches!(p, Primitive::DeviceWait) => format!(
                "{}: collectives across processes run on CUDA (NCCL); an MPS mesh is one process",
                step.label
            ),
            p => format!("{}: no Metal kernel runs {}", step.label, p.name()),
        })
    }
}

impl MpsStep {
    /// Encodes the step's kernels into the MPS stream on `inputs`, `output` and `scratch`, routing by primitive to
    /// its kernel family. Errors when the output has more elements than 32-bit kernel indices reach.
    pub(crate) fn run_raw(
        &self,
        inputs: &[*const u8],
        output: *mut u8,
        scratch: *mut u8,
        keep: Vec<Tensor>,
    ) -> Result<(), String> {
        let n = self.output.1.numel();

        if n > u32::MAX as usize {
            return Err(format!(
                "{}: {n} elements are more than MPS kernels index",
                self.primitive.name()
            ));
        }

        // Its tuned config applies to the views it was tuned for, unless an input is a view its kernel would load
        // element by element.
        let tuned = self
            .tuned
            .as_ref()
            .filter(|t| t.views == self.views && !crate::codegen::mps::loads_views(self));

        let encoding = Encoding {
            inputs: inputs.to_vec(),
            output,
            scratch,
            keep,
        };

        self.kernel.run(self, tuned, &encoding)
    }
}

/// Byte alignment of each buffer carved out of the timing scratch.
const ALIGNMENT: usize = 256;

/// Timer that runs tuning candidates on the first MPS device.
pub(crate) struct MpsTimer;

impl Timer for MpsTimer {
    type Step = MpsStep;
    type Buffers = TimerBuffers;

    fn device(&self) -> Device {
        Device::Mps(0)
    }

    /// Allocates the timing buffers for `step`.
    fn buffers(&self, step: &MpsStep) -> Result<Option<TimerBuffers>, String> {
        Ok(TimerBuffers::new(step))
    }

    /// Encodes one launch, rotating through input copies so repeats read cold memory.
    fn run(&self, buffers: &TimerBuffers, step: &MpsStep, rep: usize) -> Result<(), String> {
        let inputs = &buffers.copies[rep % buffers.copies.len()];
        step.run_raw(inputs, buffers.output, buffers.scratch, Vec::new())
    }
}

/// Input bytes to spread across copies so repeated runs miss the cache.
const COLD_BYTES: usize = 64 << 20;
/// Most input copies allocated for timing.
const MOST_COPIES: usize = 256;

/// Zeroed device memory holding rotated input copies, the outputs and scratch for a timed step.
pub(crate) struct TimerBuffers {
    _memory: Scratch,
    /// Per copy, the input pointers followed by the extra output pointers.
    copies: Vec<Vec<*const u8>>,
    output: *mut u8,
    scratch: *mut u8,
}

/// Bytes spanned by a tensor of type `ty` under `view`, including the view's offset when `offset` is set.
fn bytes_spanned(ty: &TensorType, view: Option<&View>, offset: bool) -> usize {
    let size = ty.dtype.size_of();

    match view {
        _ if ty.numel() == 0 => 0,
        None => ty.numel() * size,
        Some(view) => {
            let last: usize = ty
                .shape
                .iter()
                .zip(&view.strides)
                .map(|(&d, &s)| (d - 1) * s)
                .sum();

            (last + 1 + if offset { view.offset } else { 0 }) * size
        }
    }
}

/// Rounds `bytes` up to a nonzero multiple of `ALIGNMENT`.
fn aligned(bytes: usize) -> usize {
    bytes.div_ceil(ALIGNMENT).max(1) * ALIGNMENT
}

impl TimerBuffers {
    /// Lays out the buffers for `step`, halving the copy count until the allocation succeeds.
    fn new(step: &MpsStep) -> Option<Self> {
        let inputs: Vec<usize> = step
            .inputs
            .iter()
            .zip(&step.views)
            .map(|((b, ty), view)| match b {
                Buffer::Scalar(_) => ty.dtype.size_of(),
                _ => bytes_spanned(ty, view.as_ref(), false),
            })
            .map(aligned)
            .collect();

        let outputs: Vec<usize> = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .enumerate()
            .map(|(k, (_, ty))| {
                bytes_spanned(ty, step.output_views.get(k).and_then(Option::as_ref), true)
            })
            .map(aligned)
            .collect();

        let scratch = aligned(step.scratch.map_or(0, |(_, bytes)| bytes));
        let read: usize = inputs.iter().sum();
        let written: usize = outputs.iter().sum();
        let mut copies = COLD_BYTES.div_ceil(read).clamp(2, MOST_COPIES);

        let memory = loop {
            match Scratch::zeroed(read * copies + written + scratch) {
                Some(memory) => break memory,
                None if copies > 1 => copies /= 2,
                None => return None,
            }
        };

        // Layout: `copies` input sets back to back, then the outputs, then scratch.
        let base = memory.as_ptr();
        let mut at = read * copies;
        let mut outs = Vec::with_capacity(outputs.len());

        for size in &outputs {
            outs.push(base.wrapping_add(at));
            at += size;
        }

        let copies = (0..copies)
            .map(|c| {
                let mut at = c * read;

                let ins = inputs.iter().map(|size| {
                    let p = base.wrapping_add(at).cast_const();
                    at += size;
                    p
                });

                ins.chain(outs[1..].iter().map(|p| p.cast_const()))
                    .collect()
            })
            .collect();

        Some(TimerBuffers {
            output: outs[0],
            scratch: base.wrapping_add(at),
            _memory: memory,
            copies,
        })
    }
}
