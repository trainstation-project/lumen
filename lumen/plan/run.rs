//! Running a plan: its steps on a device, its stages on the host and device, and the custom ops it calls.

#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

use super::{Buffer, Plan, Source, Staged, Step};
#[cfg(lumen_mps_linked)]
use crate::graph::NEURAL_ENGINE;
use crate::graph::{Primitive, Scope, TensorType, View};
use crate::reference;
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
    fn drop(&mut self) {
        self.enter(&[]);
    }
}

/// Whether an MPS step's launch can be captured once and replayed with new pointers:
/// host-side work, custom calls and runtime scalar reads cannot.
#[cfg(lumen_mps_linked)]
fn replayable(step: &Step) -> bool {
    use Primitive::*;

    let launches = match &step.primitive {
        Fusion { name, .. } => !name.starts_with(crate::graph::NEURAL_ENGINE),
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
#[cfg(lumen_mps_linked)]
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

impl Plan {
    /// Runs each stage, moving values between host and device, and gathers the outputs.
    fn run_staged(
        &self,
        staged: &Staged,
        inputs: &[Tensor],
        workspace: Option<&Tensor>,
        device: Device,
    ) -> Result<Vec<Tensor>, String> {
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

    /// Runs on the device of the first input (the CPU when there are none).
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.run_on(
            inputs,
            inputs.first().map_or(Device::Cpu(0), Tensor::device),
        )
    }

    /// Runs on `device`, allocating the workspace and outputs; all inputs must already be there.
    /// On the meta device it only returns empty outputs.
    pub fn run_on(&self, inputs: &[Tensor], device: Device) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;

        if let Some(t) = inputs.iter().find(|t| t.device() != device) {
            return Err(format!(
                "inputs must be on one device, got {device} and {}",
                t.device()
            ));
        }

        let executor = self.executor(device)?;

        if device == Device::Meta(0) {
            let meta = |ty: &TensorType| {
                let options = TensorOptions::new().dtype(ty.dtype).device(Device::Meta(0));

                unsafe { Tensor::empty(&ty.shape, options) }
            };

            return Ok(self.outputs.iter().map(meta).collect());
        }

        if let Some(staged) = &self.staged {
            return self.run_staged(staged, inputs, None, device);
        }

        let inputs: Vec<Tensor> = inputs
            .iter()
            .zip(&self.inputs_at)
            .map(|(t, at)| match at {
                Buffer::Scalar(_) => t.to(Device::Cpu(0)),
                _ => dispatch_dtype!(t.dtype(), T => t.to(executor).contiguous::<T>()),
            })
            .collect();

        let options = |dtype| TensorOptions::new().dtype(dtype).device(executor);

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

        self.execute(executor, &inputs, &outputs, &workspace)?;
        Ok(outputs.into_iter().map(|t| t.to(device)).collect())
    }

    /// Runs an owned plan in a caller-provided workspace; outputs placed in the workspace are
    /// returned as views of it, so the next run overwrites them.
    pub fn run_in(&self, workspace: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;

        if let Some(staged) = &self.staged {
            return self.run_staged(staged, inputs, Some(workspace), workspace.device());
        }

        let device = workspace.device();
        let executor = self.executor(device)?;

        if executor != device || workspace.dtype() != DType::U8 || !workspace.is_contiguous() {
            return Err(format!(
                "the workspace must be contiguous uint8 on the CPU or MPS, got {}",
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

        self.execute(executor, &params, &[], workspace)?;

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

    /// Whether every step reading input `i` accepts it with `strides`, so it needn't be made
    /// contiguous first.
    fn reads_strided(&self, i: usize, strides: &[usize]) -> bool {
        if self.triton {
            let at = Buffer::Input(i);

            return !self.outputs_at.contains(&at)
                && self.steps.iter().all(|s| {
                    s.inputs.iter().all(|&(b, ref ty)| {
                        b != at
                            || ty.shape == self.inputs[i].shape
                                && crate::runtime::triton::reads_strided(s)
                    })
                });
        }

        #[cfg(lumen_mps_linked)]
        {
            let at = Buffer::Input(i);

            !self.outputs_at.contains(&at)
                && self.steps.iter().all(|s| {
                    let operands = s.inputs.iter().zip(&s.views).enumerate();
                    let mut reads = operands.filter(|(_, ((b, _), _))| *b == at);

                    reads.all(|(k, (_, view))| match &s.primitive {
                        Primitive::DotGeneral { .. } => {
                            let operands = [&s.inputs[0].1, &s.inputs[1].1];

                            view.is_none()
                                && operands[k].shape == self.inputs[i].shape
                                && crate::codegen::dot_general::mps::reads_strided(
                                    &s.primitive,
                                    operands,
                                    k,
                                    strides,
                                )
                        }
                        Primitive::Fusion { name, .. } if name.starts_with(NEURAL_ENGINE) => true,
                        Primitive::Fusion { body, .. } => {
                            view.is_none()
                                && s.inputs[k].1.shape == self.inputs[i].shape
                                && (crate::codegen::mps::fusion_reads_strided(body, k, strides)
                                    || crate::codegen::mps::loads_strided(body, k)
                                    || crate::codegen::mps::attention_reads_strided(body, strides))
                        }
                        _ => false,
                    })
                })
        }

        #[cfg(not(lumen_mps_linked))]
        {
            let _ = (i, strides);
            false
        }
    }

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

    /// The device the kernels run on for `device`, erroring on steps it can't run.
    fn executor(&self, device: Device) -> Result<Device, String> {
        let executor = if cfg!(lumen_mps_linked) && device == Device::Mps(0) {
            Device::Mps(0)
        } else if cfg!(lumen_tpu_linked) && matches!(device, Device::Tpu(_))
            || (self.triton || self.cuda) && matches!(device, Device::Cuda(_))
        {
            device
        } else {
            Device::Cpu(0)
        };

        if let Some(step) = self
            .steps
            .iter()
            .find(|s| matches!(s.primitive, Primitive::PallasCall { .. }))
            && !matches!(executor, Device::Tpu(_))
        {
            return Err(format!(
                "{}: a lumen.pallas kernel runs on a TPU, not {device}",
                step.label
            ));
        }

        if let Device::Tpu(_) = executor
            && let Some(step) = self.steps.iter().find(|s| {
                !matches!(s.primitive, Primitive::CustomCall { .. })
                    && s.inputs.iter().any(|(b, _)| matches!(b, Buffer::Scalar(_)))
            })
        {
            return Err(format!(
                "{}: runtime scalars do not run on TPU yet",
                step.label
            ));
        }

        // Metal has no float64 or float8 type for a kernel to compute or copy them in.
        if executor == Device::Mps(0) {
            let unsupported = |ty: &TensorType| {
                matches!(
                    ty.dtype,
                    DType::F64 | DType::F8E4M3FN | DType::F8E5M2 | DType::F8E4M3B11FNUZ
                )
            };

            let step = self.steps.iter().find_map(|s| {
                s.inputs
                    .iter()
                    .chain([&s.output])
                    .find(|(_, ty)| unsupported(ty))
                    .map(|(_, ty)| (s, ty.dtype))
            });

            if let Some((step, dtype)) = step {
                return Err(format!(
                    "{}: {} is not supported on MPS",
                    step.primitive.name(),
                    dtype.full_name()
                ));
            }
        }

        Ok(executor)
    }

    /// Launches every step against the given input, output and workspace tensors.
    fn execute(
        &self,
        executor: Device,
        inputs: &[Tensor],
        outputs: &[Tensor],
        workspace: &Tensor,
    ) -> Result<(), String> {
        let tensor = |buffer: Buffer| match buffer {
            Buffer::Input(i) | Buffer::Scalar(i) => &inputs[i],
            Buffer::Output(i) => &outputs[i],
            Buffer::Workspace(_) => workspace,
        };

        let pointer = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
            _ if ty.numel() == 0 => std::ptr::null_mut(),
            Buffer::Workspace(offset) => workspace.data_ptr().wrapping_add(offset),
            _ => tensor(buffer).data_ptr(),
        };

        if executor != Device::Mps(0)
            && !self.triton
            && self
                .steps
                .iter()
                .flat_map(|s| &s.views)
                .any(Option::is_some)
        {
            return Err("operands read as views run on MPS only".into());
        }

        let mut scopes = crate::profiler::is_enabled().then(OpenScopes::default);

        #[cfg(lumen_tpu_linked)]
        let batch = match executor {
            Device::Tpu(index) => Some(crate::runtime::tpu::Batch::new(index)),
            _ => None,
        };

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
                self.call_custom_op(step, op, executor, &tensor, workspace)?;
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

            // Operand pointers, offset by their views, then the extra outputs.
            let args: Vec<*const u8> = step
                .inputs
                .iter()
                .zip(&step.views)
                .map(|(s, v)| {
                    let at = v.as_ref().map_or(0, |v| v.offset * s.1.dtype.size_of());
                    pointer(s).cast_const().wrapping_add(at)
                })
                .chain(step.extra_outputs.iter().map(|s| pointer(s).cast_const()))
                .collect();

            let out = pointer(&step.output);

            let scratch = step.scratch.map_or(std::ptr::null_mut(), |(offset, _)| {
                workspace.data_ptr().wrapping_add(offset)
            });

            match executor {
                #[cfg(lumen_mps_linked)]
                Device::Mps(_) => {
                    let mut keep: Vec<Tensor> = step
                        .inputs
                        .iter()
                        .chain([&step.output])
                        .chain(&step.extra_outputs)
                        .map(|&(b, _)| tensor(b).holding())
                        .collect();

                    if step.scratch.is_some() {
                        keep.push(workspace.holding());
                    }

                    use crate::runtime::mps::{
                        Pointers, Unprepared, capture, encode, prepare, replay,
                    };

                    // Capture each step's launch once and replay it with new pointers on later runs.
                    let replayed = !viewed && !crate::profiler::is_enabled() && replayable(step);

                    let Some(cell) = replayed.then(|| {
                        let steps = self
                            .launches
                            .get_or_init(|| self.steps.iter().map(|_| OnceLock::new()).collect());

                        &steps[at]
                    }) else {
                        encode(step, &args, out, scratch, keep)?;
                        continue;
                    };

                    let sizes = step
                        .inputs
                        .iter()
                        .zip(&step.views)
                        .map(|(s, v)| extent(&s.1, v.as_ref()))
                        .chain(step.extra_outputs.iter().map(|s| extent(&s.1, None)));

                    let operands: Vec<(*const u8, usize)> =
                        args.iter().copied().zip(sizes).collect();

                    let pointers = Pointers {
                        workspace: (workspace.data_ptr().cast_const(), workspace.nbytes()),
                        operands: &operands,
                        output: (out.cast_const(), extent(&step.output.1, None)),
                    };

                    match cell.get() {
                        Some(Some(prepared)) => replay(prepared, &pointers, keep)?,
                        Some(None) => encode(step, &args, out, scratch, keep)?,
                        None => {
                            let captured =
                                capture(|| encode(step, &args, out, scratch, Vec::new()))?;

                            match prepare(captured, &pointers) {
                                Ok(prepared) => {
                                    replay(&prepared, &pointers, keep)?;
                                    let _ = cell.set(Some(prepared));
                                }
                                Err(unprepared) => {
                                    if let Unprepared::Unplaced = unprepared {
                                        let _ = cell.set(None);
                                    }

                                    encode(step, &args, out, scratch, keep)?;
                                }
                            }
                        }
                    }
                }
                #[cfg(lumen_tpu_linked)]
                Device::Tpu(index) => {
                    let _ = scratch;
                    crate::runtime::tpu::encode(index, step, &args, out)?;
                }
                Device::Cuda(_) => {
                    let _ = scratch;

                    let view = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
                        Buffer::Workspace(offset) => {
                            workspace.view_bytes(offset, ty.dtype, &ty.shape)
                        }
                        _ => tensor(buffer).view_bytes(0, ty.dtype, &ty.shape),
                    };

                    // A value placed in a larger buffer (a slice read in place, an in-place
                    // concatenate's slot) is read or written through its offset and strides there.
                    let strided = |(value, at): (&(Buffer, TensorType), &Option<View>)| match at {
                        None => view(value),
                        Some(at) => {
                            let (buffer, ty) = value;
                            let bytes = at.offset * ty.dtype.size_of();

                            let (base, start) = match *buffer {
                                Buffer::Workspace(offset) => (workspace, offset + bytes),
                                b => (tensor(b), bytes),
                            };

                            match base.is_contiguous() {
                                true => base.view_strided(start, ty.dtype, &ty.shape, &at.strides),
                                // A strided caller input, read through its own strides.
                                false => base.clone(),
                            }
                        }
                    };

                    let inputs: Vec<Tensor> =
                        step.inputs.iter().zip(&step.views).map(strided).collect();

                    let mut outputs = std::iter::once(&step.output)
                        .chain(&step.extra_outputs)
                        .zip(&step.output_views)
                        .map(strided);

                    let out = outputs.next().expect("a step's output");
                    let outs: Vec<Tensor> = outputs.collect();
                    let kernel = step
                        .kernel
                        .as_deref()
                        .and_then(|k| k.downcast_ref::<crate::codegen::triton::Kernel>());
                    let run = || match self.cuda {
                        true => crate::runtime::cuda::run(step, &inputs, &out, &outs),
                        false => crate::runtime::triton::run(kernel, step, &inputs, &out, &outs),
                    };

                    // Captured into a CUDA graph, the nodes it adds are the step's, by name.
                    #[cfg(all(feature = "python", lumen_cuda_linked))]
                    let run = || crate::runtime::triton::direct::counted(step.label, run);

                    // Profiled, the step's kernels are timed under its name (each keeps its own as `kernel`).
                    #[cfg(lumen_cupti_linked)]
                    let run = || {
                        crate::profiler::next_launch(step.label, Default::default);
                        crate::profiler::cupti::correlated(out.device(), run)
                    };

                    run()?;
                }
                _ => {
                    let _ = scratch;
                    let types: Vec<&TensorType> = step.inputs.iter().map(|(_, ty)| ty).collect();
                    let _kernel = crate::profiler::record_host_kernel(step.label);
                    crate::profiler::launched::launched(step.label, || None);

                    // Host collectives run here, synchronously, through the default group's store.
                    if let Some(ranks) = crate::distributed::collective::ranks(&step.primitive) {
                        let comm = crate::distributed::host::Host::new(ranks)?;
                        let send = args.first().copied().unwrap_or(std::ptr::null());
                        unsafe {
                            crate::distributed::collective::run(
                                &comm,
                                &step.primitive,
                                send,
                                out,
                                types.first().copied(),
                                &step.output.1,
                            )?
                        };
                        continue;
                    }
                    unsafe {
                        reference::eval_raw(&step.primitive, &args, &types, out, &step.output.1)
                    };
                }
            }
        }

        #[cfg(lumen_tpu_linked)]
        if let Some(batch) = batch {
            batch.run()?;
        }

        Ok(())
    }
}

impl Plan {
    /// Runs a custom op: copies each mutated operand into its output buffer unless it already
    /// lives there, then calls the registered kernel on the resulting tensors.
    fn call_custom_op<'a>(
        &self,
        step: &Step,
        (kernel, mutated, overlappable): (usize, &[usize], &[(usize, usize)]),
        executor: Device,
        tensor: &dyn Fn(Buffer) -> &'a Tensor,
        workspace: &Tensor,
    ) -> Result<(), String> {
        let view = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
            Buffer::Workspace(offset) => workspace.view_bytes(offset, ty.dtype, &ty.shape),
            _ => tensor(buffer).clone(),
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

                match executor {
                    _ if !src.is_contiguous() => {
                        out.copy_(src)?;
                    }
                    Device::Cuda(_) => {
                        out.copy_(src)?;
                    }
                    #[cfg(lumen_tpu_linked)]
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
        if executor == Device::Mps(0) {
            crate::stream::mps::mark();
        }

        let _ = executor;
        called.map_err(|e| format!("{}: {e}", step.label))
    }
}
