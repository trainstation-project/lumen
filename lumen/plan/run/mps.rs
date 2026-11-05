//! Running plan steps on MPS: each step's kernels encoded by primitive into the MPS stream, its launches
//! captured on its first run and replayed with new pointers on later ones.

use std::ops::{Deref, DerefMut};
use std::sync::OnceLock;

use super::Buffers;
use crate::autotune::mps::Tunable;
use crate::codegen::dot_general::mps::plan_matmul;
use crate::codegen::dynamic_slice::mps::{MAX_RANK, copy};
use crate::codegen::utils::mps::Mps;
use crate::graph::Primitive;
use crate::graph::{NEURAL_ENGINE, TensorType, View};
use crate::plan::StepBase;
use crate::plan::step::unsupported;
use crate::plan::{Buffer, Step};
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::MpsLaunch;
use crate::runtime::kernel::mps::{Pointers, Prepared, Unprepared, capture, prepare, replay};
use crate::runtime::mps::{Grid, dims_arg, element_arg, elementwise_grid, u32_arg, u64_arg};
use crate::runtime::wait::wait;
use crate::tensor::contiguous_strides;
use crate::{Device, Tensor};

/// A specialized encoder a step carries in its launch slot: `(step, inputs, output, scratch, keep)`.
pub(crate) type Encode = Box<
    dyn Fn(&StepBase, &[*const u8], *mut u8, *mut u8, Vec<Tensor>) -> Result<(), String>
        + Send
        + Sync,
>;

/// A step of Metal kernels: an elementwise fusion keeps what the autotuner needs to regenerate its kernel, and
/// each step its launches once captured.
#[derive(Debug, Clone)]
pub struct MpsStep {
    base: StepBase,
    /// What an elementwise fusion needs to regenerate its kernel under another tuning.
    pub(crate) kernel: Option<Tunable>,
    /// The step's launches, prepared for replay once captured (`None` when they can't be replayed); any change
    /// to the step drops them.
    captured: OnceLock<Option<Prepared>>,
}

impl MpsStep {
    /// `base` with the kernel codegen generated for it.
    pub(crate) fn new(base: StepBase, kernel: Option<Tunable>) -> Self {
        MpsStep {
            base,
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
        use Primitive::*;
        let step = &self.base;
        let n = step.output.1.numel();

        if n > u32::MAX as usize {
            return Err(format!(
                "{}: {n} elements are more than MPS kernels index",
                step.primitive.name()
            ));
        }

        // A step-specific (tuned) encoder applies unless an input is a view its kernel would load element by
        // element; a GEMM strides through its dot's operand views.
        if let Some(encode) = step
            .launch::<Encode>()
            .filter(|_| !crate::codegen::mps::loads_views(step))
        {
            return encode(step, inputs, output, scratch, keep);
        }

        if let ReduceSum { .. } | ReduceMax { .. } = step.primitive {
            return crate::codegen::reduction::mps::encode(step, inputs, output, scratch, keep);
        }
        if let Cumsum { .. } = step.primitive {
            return crate::codegen::scan::mps::encode(step, inputs, output, scratch, keep);
        }
        if let ScatterAdd { .. } = step.primitive {
            return crate::codegen::scatter_add::mps::encode(step, inputs, output, scratch, keep);
        }
        if let Fusion { .. } = step.primitive {
            return crate::graph_compiler::mps::encode(step, inputs, output, scratch, keep);
        }
        if let LinearCrossEntropy = step.primitive {
            return crate::codegen::dot_general::mps::encode_linear_cross_entropy(
                step, inputs, output, scratch, keep,
            );
        }

        if let Concatenate { .. } = step.primitive {
            return Err("concatenate runs in a fusion on MPS: compile with fuse".into());
        }
        if step.primitive.is_collective() || matches!(step.primitive, DeviceWait) {
            return Err(format!(
                "{}: collectives across processes run on CUDA (NCCL); an MPS mesh is one process",
                step.label
            ));
        }

        match step.primitive {
            Add
            | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
            | Unary(_)
            | Binary(_)
            | Gather { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. } => {
                unreachable!("a one-node fusion's elementwise kernel (graph_compiler::mps::alone)")
            }
            ReduceSum { .. }
            | ReduceMax { .. }
            | Cumsum { .. }
            | Concatenate { .. }
            | ScatterAdd { .. } => {
                unreachable!("encoded above")
            }
            AllReduce { .. }
            | AllGather { .. }
            | ReduceScatter { .. }
            | AllToAll { .. }
            | Broadcast { .. }
            | Reduce { .. }
            | GatherTo { .. }
            | ScatterFrom { .. }
            | Send { .. }
            | Recv { .. }
            | DeviceWait => {
                unreachable!("rejected above")
            }
            // The precompiled matmul kernel for a lone `dot_general`.
            DotGeneral { .. } => {
                let (lhs, rhs, out) = (&step.inputs[0].1, &step.inputs[1].1, &step.output.1);
                let name = step.label;

                let strides = |k: usize| {
                    let ty = &step.inputs[k].1;

                    step.views[k]
                        .as_ref()
                        .map_or_else(|| contiguous_strides(&ty.shape), |v| v.strides.clone())
                };

                let launch_ = plan_matmul(
                    &step.primitive,
                    lhs,
                    rhs,
                    out,
                    &strides(0),
                    &strides(1),
                    name,
                    None,
                )?;

                let Primitive::DotGeneral { accum_dtype, .. } = &step.primitive else {
                    unreachable!("a dot_general")
                };

                Mps::launch(MpsLaunch {
                    kernel: &format!("matmul_{}_{accum_dtype}_{}", lhs.dtype, out.dtype),
                    buffers: &[inputs[0], inputs[1], output.cast_const()],
                    args: &[dims_arg(launch_.p)],
                    grid: launch_.grid,
                    keep,
                    name,
                })
            }

            // A reshape is the copy the plan writes an output aliasing another buffer with: the input's bytes
            // copied whole, a reshape keeping their order.
            Reshape { .. } => crate::codegen::dynamic_slice::mps::copy(
                inputs[0],
                output,
                &step.output.1,
                keep,
                step.label,
            ),
            // A dynamic slice or update; an out-of-place update first copies the operand into the output.
            DynamicSlice { .. } | DynamicUpdateSlice => {
                let x = &step.inputs[0].1;
                let rank = x.shape.len();

                if rank > MAX_RANK {
                    return Err(format!(
                        "{}: {rank} dimensions are more than its MPS kernels take ({MAX_RANK})",
                        step.label
                    ));
                }

                let bytes = x.dtype.size_of();

                let (source, block, first) = match &step.primitive {
                    Primitive::DynamicSlice { slice_sizes } => (inputs[0], slice_sizes.clone(), 1),
                    Primitive::DynamicUpdateSlice => (inputs[1], step.inputs[1].1.shape.clone(), 2),
                    _ => unreachable!("a dynamic slice"),
                };

                let update = matches!(step.primitive, Primitive::DynamicUpdateSlice);

                if update && output.cast_const() != inputs[0] {
                    copy(inputs[0], output, x, keep.clone(), step.label)?;
                }

                let index_dtype = step
                    .inputs
                    .get(first)
                    .map_or("i32".into(), |(_, t)| t.dtype.to_string());

                // Unused index slots point at the source buffer; the kernel only reads the first `rank`.
                let mut buffers = vec![source, output.cast_const()];
                buffers.extend((0..MAX_RANK).map(|d| *inputs.get(first + d).unwrap_or(&source)));
                let count: usize = block.iter().product();

                let args = [
                    dims_arg(x.shape.clone()),
                    dims_arg(block.clone()),
                    u32_arg(rank as u32),
                    u64_arg(count),
                ];

                // Use 32-bit offsets when the operand fits, 64-bit otherwise.
                let width = match x.numel() <= u32::MAX as usize {
                    true => 32,
                    false => 64,
                };

                let kernel = match update {
                    true => format!("dynamic_update_slice_{bytes}_{index_dtype}_{width}"),
                    false => format!("dynamic_slice_{bytes}_{index_dtype}_{width}"),
                };

                let grid = Grid::Threads([count.div_ceil(16 / bytes), 1, 1]);
                Mps::launch(MpsLaunch {
                    kernel: &kernel,
                    buffers: &buffers,
                    args: &args,
                    grid,
                    keep,
                    name: step.label,
                })
            }
            // A host wait blocks until the first kept tensor's device is done.
            HostWait => {
                wait(&keep[0]);
                Ok(())
            }
            ToHost | ToDevice => {
                unreachable!("a stage's transfers are between its plans (compiler::stages)")
            }
            // Fill and iota: `codegen/factory/mps/kernel.metal`'s kernels.
            Full { .. } | Iota { .. } => {
                let out = &step.output.1;

                // The kernel binds the step's inputs, then its output.
                let mut buffers = inputs.to_vec();
                buffers.push(output.cast_const());

                match &step.primitive {
                    Full { fill_value, .. } => {
                        let n = out.numel();
                        let args = [element_arg(out.dtype, *fill_value), u32_arg(n as u32)];
                        let kernel = format!("fill_{}", out.dtype.size_of());
                        let grid = elementwise_grid(n, out.dtype);
                        Mps::launch(MpsLaunch {
                            kernel: &kernel,
                            buffers: &buffers,
                            args: &args,
                            grid,
                            keep,
                            name: step.label,
                        })
                    }
                    Iota { dimension, .. } => {
                        let (outer, inner) =
                            (&out.shape[..*dimension], &out.shape[dimension + 1..]);

                        // Grid is [inner, iota axis, outer] so the kernel reads its index from gid.y.
                        let grid = Grid::Threads([
                            inner.iter().product(),
                            out.shape[*dimension],
                            outer.iter().product(),
                        ]);

                        Mps::launch(MpsLaunch {
                            kernel: &format!("iota_{}", out.dtype),
                            buffers: &buffers,
                            args: &[],
                            grid,
                            keep,
                            name: step.label,
                        })
                    }
                    _ => unreachable!("a factory primitive"),
                }
            }
            // A lone `random_bits` step writing Philox words into `output`.
            RandomBits { .. } => {
                let Primitive::RandomBits { offset, .. } = step.primitive else {
                    unreachable!("a random_bits step")
                };

                let n = step.output.1.numel();
                let args = [u64_arg(offset as usize), u32_arg(n as u32)];
                let grid = elementwise_grid(n, step.output.1.dtype);

                // The kernel binds the step's inputs, then its output.
                let mut buffers = inputs.to_vec();
                buffers.push(output.cast_const());

                Mps::launch(MpsLaunch {
                    kernel: "random_bits",
                    buffers: &buffers,
                    args: &args,
                    grid,
                    keep,
                    name: step.label,
                })
            }
            Fusion { .. } | LinearCrossEntropy => unreachable!("encoded above"),
            FusionOutput { .. } => unreachable!("a fusion's kernel writes it: no step"),
            CustomCall { .. } => unreachable!("the plan calls a custom op's function itself"),
            PallasCall { .. } => unreachable!("a plan with a lumen.pallas kernel runs on a TPU"),
            TakeAlongAxis { .. } => {
                unreachable!("rewritten as a gather (passes/take_along_axis.rs)")
            }
        }
    }
}
