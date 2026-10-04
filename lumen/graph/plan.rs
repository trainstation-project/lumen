use std::cmp::Reverse;
use std::fmt;

use super::{Graph, Node, Primitive, TensorType, Var};
use crate::ops::reference;
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Device, Tensor, TensorOptions};

/// Workspace offsets are multiples of this: the device allocators'
/// alignment (`cudaMalloc`'s 256 bytes).
pub const ALIGNMENT: usize = 256;

/// Where a value lives while a plan runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    /// The caller's input `i`.
    Input(usize),
    /// Output `i`, allocated for each run and returned.
    Output(usize),
    /// The bytes at this offset in the workspace.
    Workspace(usize),
    /// The caller's input `i`, a scalar a kernel takes by value (a runtime
    /// scalar, [`PlanOptions::scalars`]): read on the host when its steps
    /// are encoded, never copied to the device.
    Scalar(usize),
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Buffer::Input(i) => write!(f, "in{i}"),
            Buffer::Output(i) => write!(f, "out{i}"),
            Buffer::Workspace(offset) => write!(f, "ws+{offset}"),
            Buffer::Scalar(i) => write!(f, "s{i}"),
        }
    }
}

/// An operand read in place as a strided view of its buffer (a slice the
/// plan does not compute, [`PlanOptions::views`]): the element offset of
/// its first element and its strides, in elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub strides: Vec<usize>,
}

/// One kernel launch, `output = primitive(inputs...)`, with each buffer's
/// type (a `reshape` with the operand's own shape is a copy).
#[derive(Debug, Clone)]
pub struct Step {
    pub primitive: Primitive,
    /// What the profiler and the graph page call it: its primitive's name,
    /// unless the compiler names it (a merged dot, `3x dot_general`).
    pub label: &'static str,
    pub inputs: Vec<(Buffer, TensorType)>,
    /// Each input's view of its buffer, if it reads one (else the whole
    /// buffer, contiguous).
    pub views: Vec<Option<View>>,
    pub output: (Buffer, TensorType),
    /// A multi-output fusion's other outputs, in order (its
    /// [`Primitive::FusionOutput`]s, which have no steps).
    pub extra_outputs: Vec<(Buffer, TensorType)>,
    /// Workspace bytes the kernel uses while it runs (its offset and
    /// length), if it asked for any ([`PlanOptions::scratch`]).
    pub scratch: Option<(usize, usize)>,
}

/// The scratch bytes a step's kernel needs while it runs, from its
/// primitive, operand types and result type.
pub type ScratchFn = fn(&Primitive, &[&TensorType], &TensorType) -> usize;

/// What a device compiler gives the planner (XLA: what a backend gives
/// buffer assignment).
#[derive(Debug, Clone, Default)]
pub struct PlanOptions {
    /// Each step's kernel scratch ([`ScratchFn`]): placed in the workspace,
    /// alive for that step alone, so kernels never allocate.
    pub scratch: Option<ScratchFn>,
    /// Inputs the caller donates (JAX: `donate_argnums`): an output of an
    /// input's type may be written into its buffer, in place, rather than
    /// into new memory, once nothing reads the input any more.
    pub donate: Vec<usize>,
    /// Plan an executable that owns its memory ([`Plan::run_in`]): the
    /// inputs not marked here and every output placed in the workspace too,
    /// the inputs copied in on each run and the outputs views of it. Those
    /// marked are parameters, whose memory is placed elsewhere (lumen's
    /// parameter store) and read in place. `None`: the inputs and outputs
    /// are the caller's ([`Plan::run`]).
    pub parameters: Option<Vec<bool>>,
    /// Slices the plan does not compute: each a view of its operand's
    /// buffer, which the steps reading it read in place at its offset and
    /// strides ([`Step::views`]). The device compiler picks them: slices
    /// read only by kernels that take strided operands.
    pub views: Vec<Var>,
    /// Inputs passed to the kernels reading them by value (one-element
    /// runtime scalars: `lumen.compile`'s floats): [`Buffer::Scalar`], no
    /// memory of their own. The device compiler picks them: inputs only
    /// kernels taking scalars by value read.
    pub scalars: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Plan {
    inputs: Vec<TensorType>,
    outputs: Vec<TensorType>,
    /// The donated input each output is written into, if any.
    aliases: Vec<Option<usize>>,
    /// Where each input and output is: in an owned plan, in the workspace
    /// (or, for a parameter, the input itself).
    inputs_at: Vec<Buffer>,
    outputs_at: Vec<Buffer>,
    steps: Vec<Step>,
    workspace_bytes: usize,
    /// Inputs a graph compiler added, each the concatenation of these
    /// parameter inputs along a dimension: one block they are placed in
    /// side by side ([`crate::Tensor::pack`]), so no step concatenates them.
    pub(crate) packed: Vec<(Vec<usize>, usize)>,
}

impl Plan {
    pub fn compile(graph: &Graph) -> Self {
        Self::compile_with(graph, &PlanOptions::default())
    }

    /// `graph`'s plan: its steps in order, each value (and kernel scratch)
    /// placed in one workspace, values whose lifetimes do not overlap
    /// sharing bytes; outputs in their own memory, or in a donated input's.
    pub fn compile_with(graph: &Graph, options: &PlanOptions) -> Self {
        let n = graph.types.len();
        let ty = |v: Var| graph.type_of(v).clone();
        let is_reshape = |p: &Primitive| matches!(p, Primitive::Reshape { .. });
        let bytes = |v: Var| graph.type_of(v).numel() * graph.type_of(v).dtype.size_of();

        let mut live = vec![false; n];
        for &v in graph.outputs() {
            live[v] = true;
        }
        for node in graph.nodes().iter().rev() {
            if live[node.output] {
                for &v in &node.inputs {
                    live[v] = true;
                }
            }
        }
        // Each value's readers (live nodes, by index).
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (t, node) in graph.nodes().iter().enumerate() {
            if live[node.output] {
                node.inputs.iter().for_each(|&v| readers[v].push(t));
            }
        }

        // A reshape's value is the bytes of its operand's root; a viewed
        // slice's, a view of them; a dynamic_update_slice's, its operand's,
        // in place, where nothing reads those after it (nor as its update
        // or an index) and they are no input's or output's (a donated
        // input's: below).
        let mut root: Vec<Var> = (0..n).collect();
        let mut view: Vec<Option<View>> = vec![None; n];
        for (t, node) in graph.nodes().iter().enumerate() {
            if is_reshape(&node.primitive) {
                root[node.output] = root[node.inputs[0]];
            }
            if let Primitive::DynamicUpdateSlice = node.primitive
                && live[node.output]
            {
                let r = root[node.inputs[0]];
                let mine = |v: Var| root[v] == r;
                let free = !graph.inputs().contains(&r)
                    && !graph.outputs().iter().any(|&o| mine(o))
                    && !node.inputs[1..].iter().any(|&v| mine(v))
                    && !(0..n).any(|v| mine(v) && readers[v].iter().any(|&u| u > t));
                if free {
                    root[node.output] = r;
                }
            }
            if let Primitive::Slice { start_indices, .. } = &node.primitive
                && options.views.contains(&node.output)
            {
                let x = node.inputs[0];
                assert!(view[x].is_none(), "a view of a view");
                let strides = contiguous_strides(&graph.type_of(x).shape);
                let offset = start_indices.iter().zip(&strides).map(|(s, t)| s * t).sum();
                root[node.output] = root[x];
                view[node.output] = Some(View { offset, strides });
            }
        }
        let is_view = |v: Var| view[v].is_some();
        let is_fusion_output = |p: &Primitive| matches!(p, Primitive::FusionOutput { .. });
        let nodes: Vec<_> = graph
            .nodes()
            .iter()
            .filter(|node| {
                live[node.output]
                    && !is_reshape(&node.primitive)
                    && !is_view(node.output)
                    && !is_fusion_output(&node.primitive)
            })
            .collect();

        let owned = options.parameters.as_ref();
        let is_parameter = |i: usize| owned.is_none_or(|p| p[i]);
        let mut buffer: Vec<Option<Buffer>> = vec![None; n];
        for (i, &v) in graph.inputs().iter().enumerate() {
            if options.scalars.contains(&i) {
                buffer[v] = Some(Buffer::Scalar(i));
            } else if is_parameter(i) {
                buffer[v] = Some(Buffer::Input(i));
            }
        }
        let mut copies = Vec::new();
        if owned.is_none() {
            for (k, &v) in graph.outputs().iter().enumerate() {
                match buffer[root[v]] {
                    None => buffer[root[v]] = Some(Buffer::Output(k)),
                    Some(_) => copies.push((k, v)),
                }
            }
        }

        // Each root's lifetime, in steps: from the first step that writes it
        // (an in-place dynamic_update_slice writes it again) to the last that
        // reads it (the copies run after every other step).
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0; n];
        for (t, node) in nodes.iter().enumerate() {
            first[root[node.output]] = first[root[node.output]].min(t);
            last[root[node.output]] = t;
            for &v in &node.inputs {
                last[root[v]] = t;
            }
        }
        for &(_, v) in &copies {
            last[root[v]] = nodes.len();
        }
        // A fusion's other outputs are written by its step, its output's
        // first; each its own value (`root`), read by later steps.
        let mut extra: Vec<Vec<(usize, Var)>> = vec![Vec::new(); n];
        for node in graph.nodes().iter().filter(|n| live[n.output]) {
            if let Primitive::FusionOutput { index, .. } = node.primitive {
                let fusion = node.inputs[0];
                first[node.output] = first[root[fusion]];
                last[node.output] = last[node.output].max(first[node.output]);
                extra[fusion].push((index, node.output));
            }
        }
        if owned.is_some() {
            // Inputs are copied in before the first step; outputs are read
            // after the last.
            for &v in graph.inputs() {
                first[v] = 0;
            }
            for &v in graph.outputs() {
                last[root[v]] = nodes.len();
            }
        }

        // Donation: an output goes into a donated input's buffer if it has
        // the input's type, and nothing reads the input after the step that
        // writes the output (which may read it only at the element it
        // writes, or as a dynamic_update_slice's operand: in place). An
        // owned plan's too: a parameter's new value, written there.
        let mut aliases = vec![None; graph.outputs().len()];
        let mut donors: Vec<usize> = options.donate.clone();
        for (k, &v) in graph.outputs().iter().enumerate() {
            let unplaced = match owned {
                None => buffer[v] == Some(Buffer::Output(k)),
                Some(_) => buffer[v].is_none(),
            };
            if v != root[v] || !unplaced {
                continue;
            }
            let fits = |&i: &usize| {
                let u = graph.inputs()[i];
                let writer = nodes[first[v]];
                let read_by_writer = writer.inputs.iter().any(|&w| root[w] == u);
                let read_later =
                    last[u] > first[v] || (read_by_writer && !reads_in_place(writer, u, &root));
                let is_output = graph.outputs().iter().any(|&o| root[o] == u);
                ty(u) == ty(v) && !read_later && !is_output
            };
            if let Some(pos) = donors.iter().position(fits) {
                let i = donors.remove(pos);
                buffer[v] = Some(Buffer::Input(i));
                aliases[k] = Some(i);
            }
        }

        // The workspace: each value without a buffer, alive from its step
        // to its last reader, and each kernel's scratch, alive for its step,
        // placed largest first at the lowest offset clear of the regions
        // alive at the same time.
        enum Region {
            Value(Var),
            Scratch(usize),
        }

        let mut regions: Vec<(usize, usize, usize, Region)> = (0..n)
            .filter(|&r| root[r] == r && buffer[r].is_none() && first[r] != usize::MAX)
            .map(|r| (bytes(r), first[r], last[r], Region::Value(r)))
            .collect();
        if let Some(scratch) = options.scratch {
            for (t, node) in nodes.iter().enumerate() {
                let types: Vec<&TensorType> =
                    node.inputs.iter().map(|&v| graph.type_of(v)).collect();
                let size = scratch(&node.primitive, &types, graph.type_of(node.output));
                if size > 0 {
                    regions.push((size, t, t, Region::Scratch(t)));
                }
            }
        }
        regions.sort_by_key(|&(size, ..)| Reverse(size));
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut scratch_at = vec![None; nodes.len()];
        let mut workspace_bytes = 0;
        for (size, from, to, region) in regions {
            let mut taken: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(_, _, f, l)| f <= to && from <= l)
                .map(|&(offset, size, _, _)| (offset, size))
                .collect();
            taken.sort();
            let mut offset = 0;
            for (start, len) in taken {
                if offset + size <= start {
                    break;
                }
                offset = offset.max((start + len).next_multiple_of(ALIGNMENT));
            }
            placed.push((offset, size, from, to));
            match region {
                Region::Value(r) => buffer[r] = Some(Buffer::Workspace(offset)),
                Region::Scratch(t) => scratch_at[t] = Some((offset, size)),
            }
            workspace_bytes = workspace_bytes.max(offset + size);
        }

        let slot = |v: Var| {
            (
                buffer[root[v]].expect("every live value has a buffer"),
                ty(v),
            )
        };
        let inputs_at = graph
            .inputs()
            .iter()
            .map(|&v| buffer[v].expect("every input has a buffer"))
            .collect();
        let outputs_at = graph
            .outputs()
            .iter()
            .enumerate()
            .map(|(k, &v)| match owned {
                Some(_) => buffer[root[v]].expect("every output has a buffer"),
                None => Buffer::Output(k),
            })
            .collect();
        let mut steps: Vec<Step> = nodes
            .iter()
            .zip(scratch_at)
            .map(|(node, scratch)| Step {
                primitive: node.primitive.clone(),
                label: graph.label(node),
                inputs: node.inputs.iter().map(|&v| slot(v)).collect(),
                views: node.inputs.iter().map(|&v| view[v].clone()).collect(),
                output: slot(node.output),
                extra_outputs: {
                    let mut outputs = extra[node.output].clone();
                    outputs.sort_unstable();
                    outputs.into_iter().map(|(_, v)| slot(v)).collect()
                },
                scratch,
            })
            .collect();
        for (k, v) in copies {
            steps.push(Step {
                primitive: Primitive::Reshape {
                    new_sizes: ty(v).shape,
                },
                label: "reshape",
                inputs: vec![slot(v)],
                views: vec![view[v].clone()],
                output: (Buffer::Output(k), ty(v)),
                extra_outputs: Vec::new(),
                scratch: None,
            });
        }
        Plan {
            inputs: graph.inputs().iter().map(|&v| ty(v)).collect(),
            outputs: graph.outputs().iter().map(|&v| ty(v)).collect(),
            aliases,
            inputs_at,
            outputs_at,
            steps,
            workspace_bytes,
            packed: Vec::new(),
        }
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The steps, for a graph compiler to name ([`Step::label`]).
    #[cfg(lumen_mps_linked)]
    pub(crate) fn steps_mut(&mut self) -> &mut [Step] {
        &mut self.steps
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// The inputs a graph compiler added after the graph's, each a block of
    /// parameters side by side: their input positions and the dimension.
    pub fn packed(&self) -> &[(Vec<usize>, usize)] {
        &self.packed
    }

    /// Run the plan on `inputs`, which must be on one device, returning
    /// its outputs on that device (the CPU if there are no inputs). On MPS
    /// the steps are Metal kernels ([`crate::ops::mps`]); elsewhere they run on
    /// the host, with device inputs copied there and the outputs back.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.run_on(inputs, inputs.first().map_or(Device::Cpu, Tensor::device))
    }

    /// [`run`](Self::run) on `device`, where the inputs must be: for plans
    /// without inputs (creating tensors) on a device. On the meta device
    /// nothing runs: the outputs are meta tensors.
    pub fn run_on(&self, inputs: &[Tensor], device: Device) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
        if let Some(t) = inputs.iter().find(|t| t.device() != device) {
            return Err(format!(
                "inputs must be on one device, got {device} and {}",
                t.device()
            ));
        }
        let executor = self.executor(device)?;
        if device == Device::Meta {
            // Nothing to compute: outputs of the right types, without data
            // (shape inference).
            let meta = |ty: &TensorType| {
                let options = TensorOptions::new().dtype(ty.dtype).device(Device::Meta);
                // SAFETY: meta tensors have no data to read.
                unsafe { Tensor::empty(&ty.shape, options) }
            };
            return Ok(self.outputs.iter().map(meta).collect());
        }
        let mut run = crate::profiler::record_op(op_name!("plan"), || self.inputs.clone());
        run.outputs(|| self.outputs.clone());
        let inputs: Vec<Tensor> = inputs
            .iter()
            .map(|t| dispatch_dtype!(t.dtype(), T => t.to(executor).contiguous::<T>()))
            .collect();
        let options = |dtype| TensorOptions::new().dtype(dtype).device(executor);
        // SAFETY: every output and workspace byte a step reads was written
        // by an earlier step, and every output is written by one.
        let workspace = unsafe { Tensor::empty(&[self.workspace_bytes], options(DType::U8)) };
        let outputs: Vec<Tensor> = self
            .outputs
            .iter()
            .zip(&self.aliases)
            .map(|(ty, alias)| match alias {
                // The donated input's storage, overwritten in place.
                Some(i) => inputs[*i].clone(),
                None => unsafe { Tensor::empty(&ty.shape, options(ty.dtype)) },
            })
            .collect();
        self.execute(executor, &inputs, &outputs, &workspace)?;
        Ok(outputs.into_iter().map(|t| t.to(device)).collect())
    }

    /// Run an owned plan ([`PlanOptions::parameters`]) in `workspace` (at
    /// least [`workspace_bytes`](Self::workspace_bytes) bytes, on the device
    /// it runs on, kept by the caller from run to run): the inputs that are
    /// not parameters copied into their places in it (from any device),
    /// the parameters read where they are (on that device). The outputs
    /// are views of the workspace (or a parameter), valid until the next
    /// run in it overwrites them.
    pub fn run_in(&self, workspace: &Tensor, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
        self.check_inputs(inputs)?;
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
        let mut run = crate::profiler::record_op(op_name!("plan"), || self.inputs.clone());
        run.outputs(|| self.outputs.clone());
        let view =
            |offset: usize, ty: &TensorType| workspace.view_bytes(offset, ty.dtype, &ty.shape);
        // A parameter is read in place: as it is if contiguous, or if every
        // step reading it takes it at its strides (a parameter packed into a
        // block another plan merged dots over is a strided view of it);
        // else a contiguous copy.
        let read = |i: usize| {
            let at = Buffer::Input(i);
            self.outputs_at.contains(&at)
                || self
                    .steps
                    .iter()
                    .any(|s| s.inputs.iter().any(|&(b, _)| b == at))
        };
        let mut params = Vec::with_capacity(inputs.len());
        for (i, ((t, ty), at)) in inputs
            .iter()
            .zip(&self.inputs)
            .zip(&self.inputs_at)
            .enumerate()
        {
            match *at {
                Buffer::Workspace(offset) => {
                    view(offset, ty).copy_(t)?;
                    params.push(t.clone());
                }
                // Read on the host as its steps are encoded.
                Buffer::Scalar(_) => params.push(t.to(Device::Cpu)),
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
        Ok(self
            .outputs_at
            .iter()
            .zip(&self.outputs)
            .map(|(at, ty)| match *at {
                Buffer::Workspace(offset) => view(offset, ty),
                Buffer::Input(i) | Buffer::Scalar(i) => params[i].clone(),
                Buffer::Output(_) => unreachable!("an owned plan has no output buffers"),
            })
            .collect())
    }

    /// Whether every step reading parameter `i` reads it in place at
    /// `strides` (dots in matmul form, on MPS), so that a strided one needs
    /// no copy.
    fn reads_strided(&self, i: usize, strides: &[usize]) -> bool {
        #[cfg(lumen_mps_linked)]
        {
            let at = Buffer::Input(i);
            !self.outputs_at.contains(&at)
                && self.steps.iter().all(|s| {
                    let operands = s.inputs.iter().zip(&s.views).enumerate();
                    let mut reads = operands.filter(|(_, ((b, _), _))| *b == at);
                    reads.all(|(k, (_, view))| match &s.primitive {
                        // Read as it is (not through a reshape: a split
                        // dot's).
                        Primitive::DotGeneral { .. } => {
                            let operands = [&s.inputs[0].1, &s.inputs[1].1];
                            view.is_none()
                                && operands[k].shape == self.inputs[i].shape
                                && crate::ops::dot_general::mps::reads_strided(
                                    &s.primitive,
                                    operands,
                                    k,
                                    strides,
                                )
                        }
                        // A contraction with its epilogue reads its dot's
                        // operands as the dot does.
                        Primitive::Fusion { body, .. } => {
                            view.is_none()
                                && s.inputs[k].1.shape == self.inputs[i].shape
                                && crate::compiler::mps::fusion_reads_strided(body, k, strides)
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

    /// Check `inputs` against the plan's input types.
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

    /// Where the steps of a run on `device` run: MPS kernels on MPS, else
    /// the host.
    fn executor(&self, device: Device) -> Result<Device, String> {
        let executor = if cfg!(lumen_mps_linked) && device == Device::Mps {
            Device::Mps
        } else {
            Device::Cpu
        };
        if executor == Device::Mps {
            let f64_step = self.steps.iter().find(|s| {
                s.inputs
                    .iter()
                    .chain([&s.output])
                    .any(|(_, ty)| ty.dtype == DType::F64)
            });
            if let Some(step) = f64_step {
                return Err(format!(
                    "{}: float64 is not supported on MPS",
                    step.primitive.name()
                ));
            }
        }
        Ok(executor)
    }

    /// Run the steps on `executor`, with `inputs` and `outputs` the
    /// buffers of [`Buffer::Input`] and [`Buffer::Output`], on it.
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
        // Null for a value with no elements, which no kernel touches.
        let pointer = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
            _ if ty.numel() == 0 => std::ptr::null_mut(),
            Buffer::Workspace(offset) => workspace.data_ptr().wrapping_add(offset),
            _ => tensor(buffer).data_ptr(),
        };
        if executor != Device::Mps
            && self
                .steps
                .iter()
                .flat_map(|s| &s.views)
                .any(Option::is_some)
        {
            return Err("operands read as views run on MPS only".into());
        }
        for step in &self.steps {
            // A strided input (a parameter [`run_in`](Self::run_in) reads in
            // place) is read as a view at its strides.
            let strided = |&(b, _): &(Buffer, TensorType)| match b {
                Buffer::Input(i) if !inputs[i].is_contiguous() => Some(i),
                _ => None,
            };
            let with_views;
            let step = if step.inputs.iter().any(|s| strided(s).is_some()) {
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
            // A view's first element is past its buffer's; a multi-output
            // fusion's other outputs follow its inputs.
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
                Device::Mps => {
                    let mut keep: Vec<Tensor> = step
                        .inputs
                        .iter()
                        .chain([&step.output])
                        .chain(&step.extra_outputs)
                        .map(|&(b, _)| tensor(b).clone())
                        .collect();
                    if step.scratch.is_some() {
                        // The kernel uses its scratch until the GPU has run it.
                        keep.push(workspace.clone());
                    }
                    crate::ops::mps::encode(step, &args, out, scratch, keep)?;
                }
                _ => {
                    // The reference evaluates each step whole: no scratch.
                    let _ = scratch;
                    let types: Vec<&TensorType> = step.inputs.iter().map(|(_, ty)| ty).collect();
                    // SAFETY: all buffers are host memory of their step
                    // types' sizes, and the inputs were written before
                    // this step.
                    unsafe {
                        reference::eval_raw(&step.primitive, &args, &types, out, &step.output.1)
                    };
                }
            }
        }
        Ok(())
    }
}

/// ```text
/// plan (workspace 1024 bytes)
///     ws+0:f32[4,16] = dot_general[...] in0 in1
///     out0:f32[4,16] = exp ws+0
/// ```
impl fmt::Display for Plan {
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

/// Whether `node` reads each operand whose root is `r` only at the element
/// it writes (an elementwise primitive, or a fusion that uses it only in
/// elementwise ones), so its output may overwrite `r`'s buffer.
fn reads_in_place(node: &Node, r: Var, root: &[Var]) -> bool {
    use Primitive::*;
    let elementwise = |p: &Primitive| {
        matches!(
            p,
            Add | Sub
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
        )
    };
    match &node.primitive {
        Fusion { body, .. } => node
            .inputs
            .iter()
            .zip(body.inputs())
            .filter(|&(&v, _)| root[v] == r)
            .all(|(_, &b)| {
                body.nodes()
                    .iter()
                    .filter(|n| n.inputs.contains(&b))
                    .all(|n| elementwise(&n.primitive))
            }),
        // Its operand alone, in place.
        DynamicUpdateSlice => node.inputs[1..].iter().all(|&v| root[v] != r),
        p => elementwise(p),
    }
}
