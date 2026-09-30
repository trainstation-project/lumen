//! Static execution plans: a [`Graph`] compiled once into a fixed list of
//! kernel launches ([`Step`]s) over buffers assigned ahead of time. Running
//! a plan binds the input pointers, allocates the outputs and one
//! workspace, and replays the steps: no tracing, dispatch or per-op
//! allocation.
//!
//! Compiling a plan:
//!
//! * **Aliasing.** Graph values are immutable and contiguous, so a
//!   `reshape` is its operand's bytes under another shape: it gets no step
//!   and shares its operand's buffer.
//! * **Dead code.** Nodes no output depends on get no step.
//! * **Outputs.** A computed output is written straight into its output
//!   buffer. An output that is an input, or whose bytes another output
//!   already holds, is copied there by a final step.
//! * **Memory planning.** Every other value lives in the workspace, from
//!   the step that writes it to the last step that reads it. Offsets are
//!   assigned greedily by size (largest first, lowest offset that no value
//!   with an overlapping lifetime uses), so values that are never live at
//!   the same time share bytes.
//!
//! Steps run on the host for now, each one a
//! [`reference`](super::reference) kernel over raw buffers, which checks
//! the planning against the reference executor; device kernels come next.

use std::cmp::Reverse;
use std::fmt;

use super::{Graph, Primitive, TensorType, Var, reference};
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Device, Tensor};

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
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Buffer::Input(i) => write!(f, "in{i}"),
            Buffer::Output(i) => write!(f, "out{i}"),
            Buffer::Workspace(offset) => write!(f, "ws+{offset}"),
        }
    }
}

/// One kernel launch, `output = primitive(inputs...)`, with each buffer's
/// type (a `reshape` with the operand's own shape is a copy).
#[derive(Debug, Clone)]
pub struct Step {
    pub primitive: Primitive,
    pub inputs: Vec<(Buffer, TensorType)>,
    pub output: (Buffer, TensorType),
}

#[derive(Debug, Clone)]
pub struct Plan {
    inputs: Vec<TensorType>,
    outputs: Vec<TensorType>,
    steps: Vec<Step>,
    workspace_bytes: usize,
}

impl Plan {
    pub fn compile(graph: &Graph) -> Self {
        let n = graph.types.len();
        let ty = |v: Var| graph.type_of(v).clone();
        let is_reshape = |p: &Primitive| matches!(p, Primitive::Reshape { .. });
        let bytes = |v: Var| graph.type_of(v).numel() * graph.type_of(v).dtype.size_of();

        // A reshape's value is the bytes of its operand's root.
        let mut root: Vec<Var> = (0..n).collect();
        for node in graph.nodes() {
            if is_reshape(&node.primitive) {
                root[node.output] = root[node.inputs[0]];
            }
        }

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
        let nodes: Vec<_> = graph
            .nodes()
            .iter()
            .filter(|node| live[node.output] && !is_reshape(&node.primitive))
            .collect();

        let mut buffer: Vec<Option<Buffer>> = vec![None; n];
        for (i, &v) in graph.inputs().iter().enumerate() {
            buffer[v] = Some(Buffer::Input(i));
        }
        let mut copies = Vec::new();
        for (k, &v) in graph.outputs().iter().enumerate() {
            match buffer[root[v]] {
                None => buffer[root[v]] = Some(Buffer::Output(k)),
                Some(_) => copies.push((k, v)),
            }
        }

        // Each root's lifetime, in steps: from the step that writes it to
        // the last that reads it (the copies run after every other step).
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0; n];
        for (t, node) in nodes.iter().enumerate() {
            first[root[node.output]] = t;
            last[root[node.output]] = t;
            for &v in &node.inputs {
                last[root[v]] = t;
            }
        }
        for &(_, v) in &copies {
            last[root[v]] = nodes.len();
        }

        let mut pending: Vec<Var> = (0..n)
            .filter(|&r| root[r] == r && buffer[r].is_none() && first[r] != usize::MAX)
            .collect();
        pending.sort_by_key(|&r| Reverse(bytes(r)));
        let mut placed: Vec<(usize, usize, Var)> = Vec::new();
        let mut workspace_bytes = 0;
        for r in pending {
            let size = bytes(r);
            let mut taken: Vec<(usize, usize)> = placed
                .iter()
                .filter(|&&(_, _, o)| first[o] <= last[r] && first[r] <= last[o])
                .map(|&(offset, size, _)| (offset, size))
                .collect();
            taken.sort();
            let mut offset = 0;
            for (start, len) in taken {
                if offset + size <= start {
                    break;
                }
                offset = offset.max((start + len).next_multiple_of(ALIGNMENT));
            }
            placed.push((offset, size, r));
            buffer[r] = Some(Buffer::Workspace(offset));
            workspace_bytes = workspace_bytes.max(offset + size);
        }

        let slot = |v: Var| {
            (
                buffer[root[v]].expect("every live value has a buffer"),
                ty(v),
            )
        };
        let mut steps: Vec<Step> = nodes
            .iter()
            .map(|node| Step {
                primitive: node.primitive.clone(),
                inputs: node.inputs.iter().map(|&v| slot(v)).collect(),
                output: slot(node.output),
            })
            .collect();
        for (k, v) in copies {
            steps.push(Step {
                primitive: Primitive::Reshape {
                    new_sizes: ty(v).shape,
                },
                inputs: vec![slot(v)],
                output: (Buffer::Output(k), ty(v)),
            });
        }
        Plan {
            inputs: graph.inputs().iter().map(|&v| ty(v)).collect(),
            outputs: graph.outputs().iter().map(|&v| ty(v)).collect(),
            steps,
            workspace_bytes,
        }
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Run the plan on `inputs`, returning its outputs on the first
    /// input's device (the CPU if it has none). The steps run on the host:
    /// device inputs are copied there and the outputs back.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, String> {
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
        let host: Vec<Tensor> = inputs
            .iter()
            .map(|t| dispatch_dtype!(t.dtype(), T => t.to(Device::Cpu).contiguous::<T>()))
            .collect();
        // SAFETY: every output and workspace byte a step reads was written
        // by an earlier step, and every output is written by one.
        let workspace = unsafe { Tensor::empty(&[self.workspace_bytes], DType::U8) };
        let outputs: Vec<Tensor> = self
            .outputs
            .iter()
            .map(|ty| unsafe { Tensor::empty(&ty.shape, ty.dtype) })
            .collect();
        let pointer = |buffer: Buffer| match buffer {
            Buffer::Input(i) => host[i].data_ptr(),
            Buffer::Output(i) => outputs[i].data_ptr(),
            Buffer::Workspace(offset) => workspace.data_ptr().wrapping_add(offset),
        };
        for step in &self.steps {
            let args: Vec<*const u8> = step
                .inputs
                .iter()
                .map(|&(b, _)| pointer(b).cast_const())
                .collect();
            let types: Vec<&TensorType> = step.inputs.iter().map(|(_, ty)| ty).collect();
            let (out, ty) = &step.output;
            // SAFETY: all buffers are host memory of their step types'
            // sizes, and the inputs were written before this step.
            unsafe { reference::eval_raw(&step.primitive, &args, &types, pointer(*out), ty) };
        }
        let device = inputs.first().map_or(Device::Cpu, Tensor::device);
        Ok(outputs.into_iter().map(|t| t.to(device)).collect())
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
            write!(f, "\n    {out}:{ty} = {}", step.primitive)?;
            for (b, _) in &step.inputs {
                write!(f, " {b}")?;
            }
        }
        Ok(())
    }
}
