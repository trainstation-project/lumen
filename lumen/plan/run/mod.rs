//! Running a plan: its steps on its device, its stages on the host and device, and the custom ops it calls. Each
//! device's step type, and how it runs on the plan's buffers, is in its own module.

pub(crate) mod cpu;
pub(crate) mod cuda;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
pub(crate) mod triton;

use super::{Buffer, DevicePlan, Plan, PlanBase, Source, Staged, StagedPlan, Step, StepBase};
use crate::graph::{Primitive, Scope, TensorType, View};
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Device, Tensor, TensorOptions};

/// Profiler ranges currently open for the scope stack of the step being run.
#[derive(Default)]
struct OpenScopes {
    scope: Scope,
    ranges: Vec<crate::profiler::RecordGuard>,
}

impl OpenScopes {
    /// Moves to `scope`, closing ranges it leaves and opening the ones it enters.
    fn enter(&mut self, scope: Scope) {
        let common = self
            .scope
            .iter()
            .zip(scope)
            .take_while(|(a, b)| a == b)
            .count();

        // Keep ranges of the shared prefix open; close the rest and open the new suffix.
        while self.ranges.len() > common {
            self.ranges.pop();
        }

        for range in &scope[common..] {
            self.ranges
                .push(crate::profiler::record_function(range.name));
        }

        self.scope = scope;
    }
}

impl Drop for OpenScopes {
    /// Closes the open ranges innermost first.
    fn drop(&mut self) {
        while self.ranges.pop().is_some() {}
    }
}

/// The tensors a plan runs on: its inputs, outputs and workspace on its device.
#[derive(Clone, Copy)]
pub struct Buffers<'a> {
    /// The device the plan runs on.
    pub(crate) device: Device,
    inputs: &'a [Tensor],
    outputs: &'a [Tensor],
    pub(crate) workspace: &'a Tensor,
    /// Whether the step being run reads a caller input through a view of its strides, set for this run.
    #[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
    pub(crate) viewed: bool,
}

impl Buffers<'_> {
    /// The tensor holding `buffer` (the workspace for a workspace offset).
    pub(crate) fn tensor(&self, buffer: Buffer) -> &Tensor {
        match buffer {
            Buffer::Input(i) | Buffer::Scalar(i) => &self.inputs[i],
            Buffer::Output(i) => &self.outputs[i],
            Buffer::Workspace(_) => self.workspace,
        }
    }

    /// Address of a value in `buffer` (null when it is empty).
    pub(crate) fn pointer(&self, &(buffer, ref ty): &(Buffer, TensorType)) -> *mut u8 {
        match buffer {
            _ if ty.numel() == 0 => std::ptr::null_mut(),
            Buffer::Workspace(offset) => self.workspace.data_ptr().wrapping_add(offset),
            _ => self.tensor(buffer).data_ptr(),
        }
    }

    /// `step`'s operand addresses, offset by their views, then its extra outputs'.
    pub(crate) fn args(&self, step: &StepBase) -> Vec<*const u8> {
        step.inputs
            .iter()
            .zip(&step.views)
            .map(|(s, v)| {
                let at = v.as_ref().map_or(0, |v| v.offset * s.1.dtype.size_of());
                self.pointer(s).cast_const().wrapping_add(at)
            })
            .chain(
                step.extra_outputs
                    .iter()
                    .map(|s| self.pointer(s).cast_const()),
            )
            .collect()
    }

    /// Address of `step`'s kernel scratch in the workspace (null when it has none).
    #[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
    pub(crate) fn scratch(&self, step: &StepBase) -> *mut u8 {
        step.scratch.map_or(std::ptr::null_mut(), |(offset, _)| {
            self.workspace.data_ptr().wrapping_add(offset)
        })
    }

    /// `step`'s operands, output and extra outputs as tensors; a value placed in a larger buffer (a slice read in
    /// place, an in-place concatenate's slot) is read or written through its offset and strides there.
    pub(crate) fn tensors(&self, step: &StepBase) -> (Vec<Tensor>, Tensor, Vec<Tensor>) {
        let view = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
            Buffer::Workspace(offset) => self.workspace.view_bytes(offset, ty.dtype, &ty.shape),
            _ => self.tensor(buffer).view_bytes(0, ty.dtype, &ty.shape),
        };

        let strided = |(value, at): (&(Buffer, TensorType), &Option<View>)| match at {
            None => view(value),
            Some(at) => {
                let (buffer, ty) = value;
                let bytes = at.offset * ty.dtype.size_of();

                let (base, start) = match *buffer {
                    Buffer::Workspace(offset) => (self.workspace, offset + bytes),
                    b => (self.tensor(b), bytes),
                };

                match base.is_contiguous() {
                    true => base.view_strided(start, ty.dtype, &ty.shape, &at.strides),
                    // A strided caller input, read through its own strides.
                    false => base.clone(),
                }
            }
        };

        let inputs = step.inputs.iter().zip(&step.views).map(strided).collect();

        let mut outputs = std::iter::once(&step.output)
            .chain(&step.extra_outputs)
            .zip(&step.output_views)
            .map(strided);

        let out = outputs.next().expect("a step's output");
        (inputs, out, outputs.collect())
    }
}

impl PlanBase {
    /// Errors unless `inputs` match the plan's input dtypes and shapes.
    fn check_inputs(&self, inputs: &[Tensor]) -> Result<(), String> {
        if inputs.len() != self.inputs.len() {
            return Err(format!(
                "the plan takes {} inputs, got {}",
                self.inputs.len(),
                inputs.len()
            ));
        }

        for (i, (t, ty)) in inputs.iter().zip(&self.inputs).enumerate() {
            if t.dtype() != ty.dtype || t.shape() != ty.shape {
                return Err(format!(
                    "input {i} must be {ty}, got {}",
                    TensorType::new(t.dtype(), t.shape())
                ));
            }
        }

        Ok(())
    }

    /// Errors unless every input is on `device`.
    fn check_device(inputs: &[Tensor], device: Device) -> Result<(), String> {
        match inputs.iter().find(|t| t.device() != device) {
            Some(t) => Err(format!(
                "inputs must be on one device, got {device} and {}",
                t.device()
            )),
            None => Ok(()),
        }
    }

    /// Empty outputs on the meta device.
    fn meta(&self) -> Vec<Tensor> {
        let meta = |ty: &TensorType| {
            let options = TensorOptions::new().dtype(ty.dtype).device(Device::Meta(0));

            unsafe { Tensor::empty(&ty.shape, options) }
        };

        self.outputs.iter().map(meta).collect()
    }
}

impl<S: Step + Clone> Plan for DevicePlan<S> {
    /// The steps in execution order.
    fn steps(&self) -> Vec<&dyn Step> {
        self.steps.iter().map(|s| s as &dyn Step).collect()
    }

    /// Runs on `device`, allocating the workspace and outputs; all inputs must already be there.
    /// On the meta device it only returns empty outputs.
    fn run_on(&self, inputs: &[Tensor], device: Device) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
        PlanBase::check_device(inputs, device)?;

        if device == Device::Meta(0) {
            return Ok(self.meta());
        }

        S::check(&self.steps, device)?;

        let inputs: Vec<Tensor> = inputs
            .iter()
            .zip(&self.inputs_at)
            .map(|(t, at)| match at {
                Buffer::Scalar(_) => t.to(Device::Cpu(0)),
                _ => dispatch_dtype!(t.dtype(), T => t.contiguous::<T>()),
            })
            .collect();

        let options = |dtype| TensorOptions::new().dtype(dtype).device(device);

        let workspace = unsafe {
            Tensor::empty_aligned(
                &[self.workspace_bytes],
                options(DType::U8),
                self.workspace_alignment,
            )
        };

        let outputs: Vec<Tensor> = self
            .outputs
            .iter()
            .zip(&self.aliases)
            .map(|(ty, alias)| match alias {
                Some(i) => inputs[*i].clone(),
                None => unsafe { Tensor::empty(&ty.shape, options(ty.dtype)) },
            })
            .collect();

        self.execute(device, &inputs, &outputs, &workspace)?;
        Ok(outputs)
    }

    /// Runs an owned plan in a caller-provided workspace; outputs placed in the workspace are
    /// returned as views of it, so the next run overwrites them.
    fn run_in(&self, workspace: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
        let device = workspace.device();
        S::check(&self.steps, device)?;

        if workspace.dtype() != DType::U8 || !workspace.is_contiguous() {
            return Err(format!(
                "the workspace must be contiguous uint8, got {}",
                workspace.dtype()
            ));
        }

        if workspace.numel() < self.workspace_bytes {
            return Err(format!(
                "the plan needs a workspace of {} bytes, got {}",
                self.workspace_bytes,
                workspace.numel()
            ));
        }

        let view =
            |offset: usize, ty: &TensorType| workspace.view_bytes(offset, ty.dtype, &ty.shape);

        // Whether any step or output reads input `i`.
        let read = |i: usize| {
            let at = Buffer::Input(i);

            self.outputs_at.contains(&at)
                || self
                    .steps
                    .iter()
                    .any(|s| s.inputs.iter().any(|&(b, _)| b == at))
        };

        let copied = |t: &Tensor, ty: &TensorType| -> Result<Tensor, String> {
            let options = TensorOptions::new().dtype(t.dtype()).device(device);
            let copy = unsafe { Tensor::empty(t.shape(), options) };
            copy.copy_(t)?;
            Ok(copy.reshape(&ty.shape))
        };

        let mut params = Vec::with_capacity(inputs.len());

        for (i, ((t, ty), at)) in inputs
            .iter()
            .zip(&self.inputs)
            .zip(&self.inputs_at)
            .enumerate()
        {
            // Arguments that sit on another device or share storage with a donated input are copied;
            // unread or stridedly readable parameters are used as given.
            let donated =
                |j: &usize| self.aliases.contains(&Some(*j)) && inputs[*j].shares_storage_with(t);

            match *at {
                Buffer::Scalar(_) => params.push(t.to(Device::Cpu(0))),
                _ if self.arguments[i]
                    && (t.device() != device || (0..inputs.len()).any(|j| donated(&j))) =>
                {
                    params.push(copied(t, ty)?)
                }
                _ if t.device() != device => {
                    return Err(format!(
                        "parameters must be on {device}, got {}",
                        t.device()
                    ));
                }
                _ if !read(i) || t.is_contiguous() || self.reads_strided(i, t.strides()) => {
                    params.push(t.clone())
                }
                _ => params.push(dispatch_dtype!(t.dtype(), T => t.contiguous::<T>())),
            }
        }

        self.execute(device, &params, &[], workspace)?;

        self.outputs_at
            .iter()
            .zip(&self.outputs)
            .map(|(at, ty)| match *at {
                Buffer::Workspace(offset) => Ok(view(offset, ty)),
                Buffer::Input(i) if self.arguments[i] => copied(&params[i], ty),
                Buffer::Input(i) | Buffer::Scalar(i) => Ok(params[i].clone()),
                Buffer::Output(_) => unreachable!("an owned plan has no output buffers"),
            })
            .collect()
    }
}

impl<S: Step + Clone> DevicePlan<S> {
    /// Whether every step reading input `i` accepts it with `strides`, so it needn't be made
    /// contiguous first.
    fn reads_strided(&self, i: usize, strides: &[usize]) -> bool {
        let at = Buffer::Input(i);

        !self.outputs_at.contains(&at)
            && self.steps.iter().all(|s| {
                let mut reads = s.inputs.iter().enumerate().filter(|(_, (b, _))| *b == at);
                reads.all(|(k, _)| s.reads_strided(k, &self.inputs[i].shape, strides))
            })
    }

    /// Launches every step against the given input, output and workspace tensors on `device`.
    fn execute(
        &self,
        device: Device,
        inputs: &[Tensor],
        outputs: &[Tensor],
        workspace: &Tensor,
    ) -> Result<(), String> {
        let mut scopes = crate::profiler::is_enabled().then(OpenScopes::default);

        let buffers = Buffers {
            device,
            inputs,
            outputs,
            workspace,
            viewed: false,
        };

        S::batched(device, &mut || {
            for (at, step) in self.steps.iter().enumerate() {
                crate::profiler::launched::step(at, step.label);

                if let Some(scopes) = &mut scopes {
                    scopes.enter(step.scope);
                }

                if let Primitive::CustomCall {
                    kernel,
                    mutated,
                    overlappable,
                    ..
                } = &step.primitive
                {
                    let op = (*kernel, mutated.as_slice(), overlappable.as_slice());
                    call_custom_op(step, op, &buffers)?;
                    continue;
                }

                // Non-contiguous inputs are read through views of their strides.
                let strided = |&(b, _): &(Buffer, TensorType)| match b {
                    Buffer::Input(i) if !inputs[i].is_contiguous() => Some(i),
                    _ => None,
                };

                let with_views;
                let viewed = step.inputs.iter().any(|s| strided(s).is_some());

                let step = if viewed {
                    let mut s = step.clone();

                    for (k, input) in step.inputs.iter().enumerate() {
                        if let Some(i) = strided(input) {
                            let strides = inputs[i].strides().to_vec();
                            s.views[k] = Some(View { offset: 0, strides });
                        }
                    }

                    with_views = s;
                    &with_views
                } else {
                    step
                };

                let mut record = crate::profiler::record_op(step.label, || {
                    step.inputs.iter().map(|(_, ty)| ty.clone()).collect()
                });

                record.outputs(|| {
                    let extra = step.extra_outputs.iter().map(|(_, ty)| ty.clone());

                    std::iter::once(step.output.1.clone())
                        .chain(extra)
                        .collect()
                });

                record.accum(|| step.primitive.accum_dtypes(step.output.1.dtype));
                record.strides(|| (step.input_strides(), step.output_strides()));
                step.run(&Buffers { viewed, ..buffers })?;
            }

            Ok(())
        })
    }
}

/// Runs a custom op: copies each mutated operand into its output buffer unless it already
/// lives there, then calls the registered kernel on the resulting tensors.
fn call_custom_op(
    step: &StepBase,
    (kernel, mutated, overlappable): (usize, &[usize], &[(usize, usize)]),
    buffers: &Buffers,
) -> Result<(), String> {
    let view = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
        Buffer::Workspace(offset) => buffers.workspace.view_bytes(offset, ty.dtype, &ty.shape),
        _ => buffers.tensor(buffer).clone(),
    };

    let outputs: Vec<&(Buffer, TensorType)> = std::iter::once(&step.output)
        .chain(&step.extra_outputs)
        .collect();

    let mut args: Vec<Tensor> = step.inputs.iter().map(view).collect();

    for (k, &m) in mutated.iter().enumerate() {
        let out = view(outputs[k]);
        let src = &args[m];

        let overlapped = overlappable
            .iter()
            .any(|&(o, j)| o == m && args[j].data_ptr() == out.data_ptr());

        if out.data_ptr() != src.data_ptr() && !overlapped {
            let ty = &outputs[k].1;

            match buffers.device {
                _ if !src.is_contiguous() => {
                    out.copy_(src)?;
                }
                Device::Cuda(_) => {
                    out.copy_(src)?;
                }
                Device::Tpu(_) => {
                    return Err(format!("{}: custom ops do not run on TPU yet", step.label));
                }
                #[cfg(lumen_mps_linked)]
                Device::Mps(_) => crate::codegen::dynamic_slice::mps::copy(
                    src.data_ptr().cast_const(),
                    out.data_ptr(),
                    ty,
                    vec![src.clone(), out.clone()],
                    "copy",
                )?,
                _ => unsafe {
                    let bytes = ty.numel() * ty.dtype.size_of();
                    std::ptr::copy_nonoverlapping(src.data_ptr(), out.data_ptr(), bytes);
                },
            }
        }

        args[m] = out;
    }

    let called = crate::graph::custom::call(kernel, &args);

    #[cfg(lumen_mps_linked)]
    if buffers.device == Device::Mps(0) {
        crate::stream::mps::mark();
    }

    called.map_err(|e| format!("{}: {e}", step.label))
}

impl Plan for StagedPlan {
    /// Every stage's steps, in turn.
    fn steps(&self) -> Vec<&dyn Step> {
        self.staged
            .stages
            .iter()
            .flat_map(|s| s.plan.steps())
            .collect()
    }

    /// The host and device stages.
    fn staged(&self) -> Option<&Staged> {
        Some(&self.staged)
    }

    /// Runs each stage on `device` or the host, the device stages in workspaces of their own.
    fn run_on(&self, inputs: &[Tensor], device: Device) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
        PlanBase::check_device(inputs, device)?;

        match device {
            Device::Meta(_) => Ok(self.meta()),
            _ => self.run_staged(inputs, None, device),
        }
    }

    /// Runs each stage, the device stages in their slices of `workspace`.
    fn run_in(&self, workspace: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
        self.run_staged(inputs, Some(workspace), workspace.device())
    }
}

impl StagedPlan {
    /// Runs each stage, moving values between host and device, and gathers the outputs.
    fn run_staged(
        &self,
        inputs: &[Tensor],
        workspace: Option<&Tensor>,
        device: Device,
    ) -> Result<Vec<Tensor>, String> {
        let staged = &self.staged;
        // Program inputs come first; packed blocks follow them.
        let n = self.inputs.len() - self.packed.len();
        let mut values: Vec<Vec<Tensor>> = Vec::with_capacity(staged.stages.len());

        let resolve = |values: &[Vec<Tensor>], source: Source| match source {
            Source::Input(i) => inputs[i].clone(),
            Source::Stage(s, k) => values[s][k].clone(),
            Source::ToHost(s, k) => {
                let value = &values[s][k];
                let mut op = crate::profiler::record_op("to_host", || vec![value.ty()]);
                op.outputs(|| vec![value.ty()]);
                value.to(Device::Cpu(0))
            }
            Source::ToDevice(s, k) => {
                let value = &values[s][k];
                let mut op = crate::profiler::record_op("to_device", || vec![value.ty()]);
                op.outputs(|| vec![value.ty()]);
                value.to(device)
            }
        };

        for stage in &staged.stages {
            let mut ins: Vec<Tensor> = stage.inputs.iter().map(|&s| resolve(&values, s)).collect();
            ins.extend(stage.blocks.iter().map(|&b| inputs[n + b].clone()));

            // Host stages run on CPU copies; device stages run in their slice of the shared
            // workspace when there is one.
            let outputs = match (stage.host, workspace) {
                (true, _) => {
                    let ins: Vec<Tensor> = ins.iter().map(|t| t.to(Device::Cpu(0))).collect();
                    stage.plan.run_on(&ins, Device::Cpu(0))?
                }
                (false, Some(ws)) => {
                    let bytes = stage.plan.workspace_bytes();
                    let slice = ws.view_bytes(stage.offset, DType::U8, &[bytes]);
                    stage.plan.run_in(&slice, &ins)?
                }
                (false, None) => {
                    let ins: Vec<Tensor> = ins.iter().map(|t| t.to(device)).collect();
                    stage.plan.run_on(&ins, device)?
                }
            };

            values.push(outputs);
        }

        Ok(staged
            .outputs
            .iter()
            .map(|&s| match s {
                Source::Input(i) => inputs[i].to(device),
                s => resolve(&values, s),
            })
            .collect())
    }
}
