//! Lowers a graph to a `Plan`: an ordered list of kernel steps whose values are placed in
//! caller inputs, fresh outputs or one shared workspace, and runs it on a device.
use std::any::Any;
use std::cmp::Reverse;
use std::fmt;
use std::sync::Arc;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

pub use crate::graph::View;
use crate::graph::{Graph, NEURAL_ENGINE, Node, Primitive, Scope, TensorType, UPDATES, Var};
use crate::kernels::reference;
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Device, Tensor, TensorOptions};

/// Byte alignment of every value and scratch region in the workspace.
pub const ALIGNMENT: usize = 256;

/// Where a step reads or writes a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    /// The caller's input `i`, or a donated input that holds an output.
    Input(usize),
    /// The freshly allocated output `i`.
    Output(usize),
    /// A byte offset into the shared workspace.
    Workspace(usize),
    /// Input `i` passed as a host-side scalar.
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

/// One kernel launch: a primitive with the buffers it reads and writes.
#[derive(Debug, Clone)]
pub struct Step {
    pub primitive: Primitive,
    /// Name shown in profiles and error messages.
    pub label: &'static str,
    /// Operand buffers with their types.
    pub inputs: Vec<(Buffer, TensorType)>,
    /// Per operand, a strided view of its buffer, or `None` when it is read contiguously.
    pub views: Vec<Option<View>>,
    /// The buffer and type the step writes.
    pub output: (Buffer, TensorType),
    /// Further outputs of a multi-output fusion or custom call, in output index order.
    pub extra_outputs: Vec<(Buffer, TensorType)>,
    /// Per output (main output first), where it is written inside a larger buffer, if anywhere.
    pub output_views: Vec<Option<View>>,
    /// Workspace `(offset, bytes)` reserved for the kernel's own scratch.
    pub scratch: Option<(usize, usize)>,
    /// Profiler scope stack the step was traced under.
    pub scope: Scope,
    /// Python source locations `(file, line)` the step was traced from.
    pub sources: Vec<(&'static str, u32)>,
    /// The autotuned kernel choice, if one was made.
    pub tuned: Option<Tuned>,
    /// What a backend prepared to launch the step, per layout of the tensors it ran on.
    pub memo: Memo,
}

/// A step's launch preparations (generated helpers, compiled kernels), each kept under a key a backend
/// derives from what it was prepared for (its tensors' layouts), so a step that runs again on the same
/// layouts skips regenerating them. Clones of a step share it.
#[derive(Clone, Default)]
pub struct Memo(Arc<std::sync::Mutex<std::collections::HashMap<u64, Arc<dyn Any + Send + Sync>>>>);

impl Memo {
    /// The preparation kept under `key`, made by `make` and kept on first use; a failure isn't kept.
    pub(crate) fn get<T: Send + Sync + 'static>(
        &self,
        key: u64,
        make: impl FnOnce() -> Result<T, String>,
    ) -> Result<Arc<T>, String> {
        let kept = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned();
        if let Some(t) = kept.and_then(|k| k.downcast::<T>().ok()) {
            return Ok(t);
        }
        let made = Arc::new(make()?);
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, made.clone());
        Ok(made)
    }
}

impl fmt::Debug for Memo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Memo")
    }
}

/// A kernel picked by autotuning, with the backend-specific launch parameters it needs.
#[derive(Clone)]
pub struct Tuned {
    /// Name of the chosen kernel.
    pub kernel: String,
    /// Backend launch configuration, recovered with `Step::launch`.
    launch: Arc<dyn Any + Send + Sync>,
}

impl Tuned {
    /// Wraps a backend's launch configuration under the kernel name.
    pub(crate) fn new<L: Send + Sync + 'static>(kernel: String, launch: L) -> Self {
        Tuned {
            kernel,
            launch: Arc::new(launch),
        }
    }
}

impl fmt::Debug for Tuned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.kernel)
    }
}

impl Step {
    /// The tuned launch configuration, if one of type `L` was stored.
    pub(crate) fn launch<L: 'static>(&self) -> Option<&L> {
        self.tuned.as_ref()?.launch.downcast_ref::<L>()
    }

    /// Element strides each operand is read with: its view's, or contiguous ones.
    pub(crate) fn input_strides(&self) -> Vec<Vec<usize>> {
        let strides = |(_, ty): &(Buffer, TensorType), view: &Option<View>| match view {
            Some(view) => view.strides.clone(),
            None => contiguous_strides(&ty.shape),
        };
        self.inputs
            .iter()
            .zip(&self.views)
            .map(|(i, v)| strides(i, v))
            .collect()
    }

    /// Element strides each output is written with: its view's, or contiguous ones.
    pub(crate) fn output_strides(&self) -> Vec<Vec<usize>> {
        std::iter::once(&self.output)
            .chain(&self.extra_outputs)
            .zip(&self.output_views)
            .map(|((_, ty), view)| match view {
                Some(view) => view.strides.clone(),
                None => contiguous_strides(&ty.shape),
            })
            .collect()
    }
}

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
    pub(crate) triton: Option<Vec<Option<super::triton::Kernel>>>,
    /// MPS launches captured per step, replayed on later runs.
    #[cfg(lumen_mps_linked)]
    launches: Launches,
}

/// Per step, the prepared MPS launch once captured (`None` when it can't be replayed).
#[cfg(lumen_mps_linked)]
type Launches = OnceLock<Vec<OnceLock<Option<crate::kernels::mps::Prepared>>>>;

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
    /// Compiles `graph` with default options.
    pub fn compile(graph: &Graph) -> Self {
        Self::compile_with(graph, &PlanOptions::default())
    }

    /// Compiles `graph`, keeping whichever schedule needs the smallest workspace and
    /// rematerializing values when that still exceeds `options.memory_limit`.
    pub fn compile_with(graph: &Graph, options: &PlanOptions) -> Self {
        let mut best = (graph.clone(), Self::compile_in_order(graph, options));
        // Try the traced order plus list and DFS schedules.
        for order in [super::schedule::list(graph), super::schedule::dfs(graph)] {
            let scheduled = graph.reordered(&order);
            let plan = Self::compile_in_order(&scheduled, options);
            if plan.workspace_bytes < best.1.workspace_bytes {
                best = (scheduled, plan);
            }
        }
        let (scheduled, plan) = best;
        match options.memory_limit {
            Some(limit) if plan.workspace_bytes > limit => {
                super::remat::rematerialize(&scheduled, limit, |g| {
                    Self::compile_in_order(g, options)
                })
            }
            _ => plan,
        }
    }

    /// Compiles `graph` in its node order: aliases values onto shared buffers, assigns
    /// inputs and outputs, and packs the rest into the workspace by live range.
    fn compile_in_order(graph: &Graph, options: &PlanOptions) -> Self {
        let graph = &graph.reordered(&super::schedule::overlap(graph));
        let n = graph.types.len();
        let ty = |v: Var| graph.type_of(v).clone();
        let is_reshape = |node: &Node| {
            matches!(node.primitive, Primitive::Reshape { .. })
                && options.layout.is_none_or(|l| {
                    (l.same)(graph.type_of(node.inputs[0]), graph.type_of(node.output))
                })
        };
        let bytes = |v: Var| match options.layout {
            Some(l) => (l.bytes)(graph.type_of(v)),
            None => graph.type_of(v).numel() * graph.type_of(v).dtype.size_of(),
        };

        // Liveness: values the outputs depend on.
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
        // What each node reads: a `device_wait` also reads its collective's operands, which the collective
        // reads asynchronously until the wait.
        let mut producer: Vec<Option<&Node>> = vec![None; n];
        for node in graph.nodes() {
            producer[node.output] = Some(node);
        }
        let reads = |node: &Node| -> Vec<Var> {
            let mut vars = node.inputs.clone();
            if let Primitive::DeviceWait = node.primitive
                && let Some(collective) = producer[node.inputs[0]]
                && collective.primitive.is_collective()
            {
                vars.extend(&collective.inputs);
            }
            vars
        };
        // For each value, the indices of the live nodes that read it.
        let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (t, node) in graph.nodes().iter().enumerate() {
            if live[node.output] {
                reads(node).into_iter().for_each(|v| readers[v].push(t));
            }
        }

        let callers = |r: Var| graph.inputs().contains(&r);
        let placements = graph.placements();
        let in_place = |node: &Node| {
            matches!(
                node.primitive,
                Primitive::Concatenate { in_place: true, .. }
            )
        };
        // Operands of in-place concatenates are written straight into the result's buffer.
        let mut placed = vec![false; n];
        for node in graph.nodes().iter().filter(|node| in_place(node)) {
            node.inputs.iter().for_each(|&v| placed[v] = true);
        }
        // `root[v]` is the value whose buffer `v` lives in; reshapes, waits and in-place writes alias it.
        let mut root: Vec<Var> = (0..n).collect();
        let mut view: Vec<Option<View>> = vec![None; n];
        let mut written_at: Vec<Option<View>> = vec![None; n];
        for (t, node) in graph.nodes().iter().enumerate() {
            if is_reshape(node)
                || matches!(node.primitive, Primitive::HostWait | Primitive::DeviceWait)
            {
                root[node.output] = root[node.inputs[0]];
            }
            // Outputs an updating op writes over one of its operands, with that operand's position.
            let written: Vec<(Var, usize)> = match &node.primitive {
                p if updates_operand(p) => vec![(node.output, 0)],
                Primitive::CustomCall { mutated, .. } => {
                    let mut written = vec![(node.output, mutated[0])];
                    for other in graph.nodes() {
                        if let Primitive::FusionOutput { index, .. } = other.primitive
                            && other.inputs[0] == node.output
                        {
                            written.push((other.output, mutated[index]));
                        }
                    }
                    written
                }
                _ => Vec::new(),
            };
            let overlappable: &[(usize, usize)] = match &node.primitive {
                Primitive::CustomCall { overlappable, .. } => overlappable,
                _ => &[],
            };
            for (out, position) in written {
                if !live[out] {
                    continue;
                }
                // Reuse the mutated operand's buffer, or one it may overlap, if nothing else needs it.
                let candidates = std::iter::once(position).chain(
                    overlappable
                        .iter()
                        .filter(|&&(m, _)| m == position)
                        .map(|&(_, j)| j),
                );
                for j in candidates {
                    let r = root[node.inputs[j]];
                    let mine = |v: Var| root[v] == r;
                    let others = node
                        .inputs
                        .iter()
                        .enumerate()
                        .any(|(i, &v)| i != j && i != position && mine(v));
                    let free = graph.type_of(node.inputs[j]) == graph.type_of(out)
                        && !callers(r)
                        && !graph.outputs().iter().any(|&o| mine(o))
                        && !others
                        && (j == position || root[node.inputs[position]] != r)
                        && !(0..n).any(|v| mine(v) && readers[v].iter().any(|&u| u > t));
                    if free {
                        root[out] = r;
                        break;
                    }
                }
            }
            // Whether this node becomes a kernel step that writes its own output buffer.
            let neural_engine = matches!(&node.primitive, Primitive::Fusion { name, .. } if name.starts_with(NEURAL_ENGINE));
            let step = !neural_engine
                && !in_place(node)
                && !placed[node.output]
                && !updates_operand(&node.primitive)
                && !matches!(
                    node.primitive,
                    Primitive::CustomCall { .. }
                        | Primitive::Reshape { .. }
                        | Primitive::HostWait
                        | Primitive::DeviceWait
                        | Primitive::FusionOutput { .. }
                );
            // Donated outputs are placed later, in the donor's buffer.
            let donated = graph.outputs().iter().enumerate().any(|(k, &o)| {
                o == node.output
                    && (options
                        .donate
                        .iter()
                        .any(|&i| graph.type_of(graph.inputs()[i]) == graph.type_of(o))
                        || options.donate_into.iter().any(|&(_, j)| j == k))
            });
            // Write over an operand whose buffer dies here, when the kernel reads it elementwise.
            if step && live[node.output] && !donated {
                for &u in &node.inputs {
                    let r = root[u];
                    let mine = |v: Var| root[v] == r;
                    let free = graph.type_of(u) == graph.type_of(node.output)
                        && !callers(r)
                        && !graph.outputs().iter().any(|&o| mine(o))
                        && !(0..n).any(|v| mine(v) && readers[v].iter().any(|&w| w > t))
                        && reads_in_place(graph, node, node.output, r, &root);
                    if free {
                        root[node.output] = r;
                        break;
                    }
                }
            }
            if in_place(node) {
                for &v in &node.inputs {
                    assert!(
                        root[v] == v,
                        "an in-place concatenate's operand in its own memory"
                    );
                    root[v] = root[node.output];
                    written_at[v] = placements[v].as_ref().map(|(_, at)| at.clone());
                }
            }
            // A view slice reads its operand's buffer at an offset instead of copying.
            if let Primitive::Slice { view: true, .. } = node.primitive {
                let x = node.inputs[0];
                assert!(view[x].is_none(), "a view of a view");
                root[node.output] = root[x];
                view[node.output] = placements[node.output].as_ref().map(|(_, at)| at.clone());
            }
        }
        let is_view = |v: Var| view[v].is_some();
        let is_fusion_output = |p: &Primitive| matches!(p, Primitive::FusionOutput { .. });
        // The nodes that become steps.
        let nodes: Vec<_> = graph
            .nodes()
            .iter()
            .filter(|node| {
                live[node.output]
                    && !is_reshape(node)
                    && !is_view(node.output)
                    && !is_fusion_output(&node.primitive)
                    && !in_place(node)
            })
            .collect();

        // Inputs live in caller buffers; outputs of non-owned plans in fresh ones, and an
        // output whose buffer is already taken gets a copy step.
        let owned = options.parameters.as_ref();
        let mut buffer: Vec<Option<Buffer>> = vec![None; n];
        for (i, &v) in graph.inputs().iter().enumerate() {
            buffer[v] = Some(match options.scalars.contains(&i) {
                true => Buffer::Scalar(i),
                false => Buffer::Input(i),
            });
        }
        let arguments = (0..graph.inputs().len())
            .map(|i| owned.is_some_and(|p| !p[i]) && !options.scalars.contains(&i))
            .collect();
        let mut copies = Vec::new();
        if owned.is_none() {
            for (k, &v) in graph.outputs().iter().enumerate() {
                match buffer[root[v]] {
                    None => buffer[root[v]] = Some(Buffer::Output(k)),
                    Some(_) => copies.push((k, v)),
                }
            }
        }

        // Live range of each buffer root, in step indices.
        let mut first = vec![usize::MAX; n];
        let mut last = vec![0; n];
        for (t, node) in nodes.iter().enumerate() {
            first[root[node.output]] = first[root[node.output]].min(t);
            last[root[node.output]] = t;
            for v in reads(node) {
                last[root[v]] = t;
            }
        }
        for &(_, v) in &copies {
            last[root[v]] = nodes.len();
        }
        // Extra outputs of multi-output fusions are written by the fusion's step.
        let mut extra: Vec<Vec<(usize, Var)>> = vec![Vec::new(); n];
        for node in graph.nodes().iter().filter(|n| live[n.output]) {
            if let Primitive::FusionOutput { index, .. } = node.primitive {
                let fusion = node.inputs[0];
                let r = root[node.output];
                first[r] = first[r].min(first[root[fusion]]);
                last[r] = last[r].max(first[r]);
                extra[fusion].push((index, node.output));
            }
        }
        if owned.is_some() {
            for &v in graph.outputs() {
                last[root[v]] = nodes.len();
            }
        }

        // Place outputs in donated inputs that are not read after the output is first written.
        let mut aliases = vec![None; graph.outputs().len()];
        let paired = |i: usize| options.donate_into.iter().any(|&(j, _)| j == i);
        let mut donors: Vec<usize> = options.donate.clone();
        donors.extend(options.donate_into.iter().map(|&(i, _)| i));
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
                let block_read_later = options.blocks.iter().any(|(b, members)| {
                    let b = graph.inputs()[*b];
                    members.contains(&i) && !readers[b].is_empty() && last[root[b]] >= first[v]
                });
                let read_later = last[u] > first[v]
                    || block_read_later
                    || (read_by_writer && !reads_in_place(graph, writer, v, u, &root));
                let is_output = graph.outputs().iter().any(|&o| root[o] == u);
                let its = !paired(i) || options.donate_into.contains(&(i, k));
                ty(u) == ty(v) && !read_later && !is_output && its
            };
            if let Some(pos) = donors.iter().position(fits) {
                let i = donors.remove(pos);
                buffer[v] = Some(Buffer::Input(i));
                aliases[k] = Some(i);
            }
        }

        /// Something that needs room in the workspace.
        enum Region {
            Value(Var),
            Scratch(usize),
        }

        let align = |v: Var| match options.layout {
            Some(l) => (l.align)(graph.type_of(v)).max(ALIGNMENT),
            None => ALIGNMENT,
        };
        // Workspace regions: `(bytes, first step, last step, what)`.
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
        // Largest first, each at the lowest aligned offset that doesn't overlap a placed
        // region whose live range intersects its own.
        regions.sort_by_key(|&(size, ..)| Reverse(size));
        let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut scratch_at = vec![None; nodes.len()];
        let mut workspace_bytes = 0;
        let mut workspace_alignment = ALIGNMENT;
        for (size, from, to, region) in regions {
            let alignment = match region {
                Region::Value(r) => align(r),
                Region::Scratch(_) => ALIGNMENT,
            };
            workspace_alignment = lcm(workspace_alignment, alignment);
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
                offset = offset.max((start + len).next_multiple_of(alignment));
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
                output_views: {
                    let mut outputs = extra[node.output].clone();
                    outputs.sort_unstable();
                    std::iter::once(node.output)
                        .chain(outputs.into_iter().map(|(_, v)| v))
                        .map(|v| written_at[v].clone())
                        .collect()
                },
                scratch,
                scope: node.scope,
                sources: {
                    let mut sources = Vec::new();
                    sources_of(node, &mut sources);
                    sources
                },
                tuned: None,
                memo: Memo::default(),
            })
            .collect();
        // Outputs that alias another buffer are copied out at the end.
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
                output_views: vec![None],
                scratch: None,
                scope: &[],
                sources: Vec::new(),
                tuned: None,
                memo: Memo::default(),
            });
        }
        Plan {
            inputs: graph.inputs().iter().map(|&v| ty(v)).collect(),
            outputs: graph.outputs().iter().map(|&v| ty(v)).collect(),
            aliases,
            inputs_at,
            arguments,
            outputs_at,
            steps,
            workspace_bytes,
            workspace_alignment,
            packed: Vec::new(),
            interleaved: Vec::new(),
            staged: None,
            triton: None,
            #[cfg(lumen_mps_linked)]
            launches: OnceLock::new(),
        }
    }

    /// Builds a staged plan whose stages run in order on the host or device.
    pub(crate) fn staged(
        inputs: Vec<TensorType>,
        outputs: Vec<TensorType>,
        staged: Staged,
        steps: Vec<Step>,
        (workspace_bytes, workspace_alignment): (usize, usize),
        packed: Vec<(Vec<usize>, usize)>,
        interleaved: Vec<Vec<usize>>,
    ) -> Plan {
        Plan {
            aliases: vec![None; outputs.len()],
            inputs,
            outputs,
            inputs_at: Vec::new(),
            arguments: Vec::new(),
            outputs_at: Vec::new(),
            steps,
            workspace_bytes,
            workspace_alignment,
            packed,
            interleaved,
            staged: Some(Box::new(staged)),
            triton: None,
            #[cfg(lumen_mps_linked)]
            launches: OnceLock::new(),
        }
    }

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
        if self.triton.is_some() {
            let at = Buffer::Input(i);
            return !self.outputs_at.contains(&at)
                && self.steps.iter().all(|s| {
                    s.inputs.iter().all(|&(b, ref ty)| {
                        b != at
                            || ty.shape == self.inputs[i].shape && super::triton::reads_strided(s)
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
                                && crate::kernels::dot_general::mps::reads_strided(
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
                                && (crate::graph_compiler::mps::fusion_reads_strided(
                                    body, k, strides,
                                ) || crate::graph_compiler::mps::loads_strided(body, k)
                                    || crate::graph_compiler::mps::attention_reads_strided(
                                        body, strides,
                                    ))
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
            || self.triton.is_some() && matches!(device, Device::Cuda(_))
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
        if executor == Device::Mps(0) {
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
            && self.triton.is_none()
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
            Device::Tpu(index) => Some(crate::kernels::tpu::Batch::new(index)),
            _ => None,
        };
        for (at, step) in self.steps.iter().enumerate() {
            crate::kernels::launched::step(at, step.label);
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
                    use crate::kernels::mps::{
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
                    crate::kernels::tpu::encode(index, step, &args, out)?;
                }
                Device::Cuda(_) => {
                    let _ = scratch;
                    let view = |&(buffer, ref ty): &(Buffer, TensorType)| match buffer {
                        Buffer::Workspace(offset) => {
                            workspace.view_bytes(offset, ty.dtype, &ty.shape)
                        }
                        _ => tensor(buffer).view_bytes(0, ty.dtype, &ty.shape),
                    };
                    let kernels = self.triton.as_deref().unwrap_or_default();
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
                    let kernel = kernels.get(at).and_then(Option::as_ref);
                    let run = || super::triton::run(kernel, step, &inputs, &out, &outs);
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
                    crate::kernels::launched::launched(step.label, || None);
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
                    Device::Mps(_) => crate::kernels::dynamic_slice::mps::copy(
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

/// Whether the primitive writes its result over its first operand.
fn updates_operand(p: &Primitive) -> bool {
    match p {
        Primitive::DynamicUpdateSlice | Primitive::ScatterAdd { .. } => true,
        Primitive::Fusion { name, .. } => name.starts_with(UPDATES),
        _ => false,
    }
}

/// Whether `node` can write `out` into buffer `r` while reading it: every read of `r`
/// happens at the element being written.
fn reads_in_place(graph: &Graph, node: &Node, out: Var, r: Var, root: &[Var]) -> bool {
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
        p if updates_operand(p) => node.inputs[1..].iter().all(|&v| root[v] != r),
        Fusion { body, .. } => node
            .inputs
            .iter()
            .zip(body.inputs())
            .filter(|&(&v, _)| root[v] == r)
            // Every use of the operand inside the body must be elementwise.
            .all(|(_, &b)| {
                let (mut stack, mut seen) = (vec![b], vec![false; body.types.len()]);
                while let Some(v) = stack.pop() {
                    for n in body.nodes().iter().filter(|n| n.inputs.contains(&v)) {
                        if !elementwise(&n.primitive) {
                            return false;
                        }
                        if !std::mem::replace(&mut seen[n.output], true) {
                            stack.push(n.output);
                        }
                    }
                }
                true
            }),
        CustomCall { mutated, .. } => {
            let k = graph.nodes().iter().find_map(|n| match n.primitive {
                FusionOutput { index, .. } if n.output == out && n.inputs[0] == node.output => {
                    Some(index)
                }
                _ => None,
            });
            // Only the mutated operand may live in `r`.
            let position = mutated[k.unwrap_or(0)];
            node.inputs
                .iter()
                .enumerate()
                .all(|(j, &v)| (root[v] == r) == (j == position))
        }
        p => elementwise(p),
    }
}

/// Appends the unique source locations of `node` and, for fusions, of its body.
fn sources_of(node: &Node, sources: &mut Vec<(&'static str, u32)>) {
    for &s in node.sources {
        if !sources.contains(&s) {
            sources.push(s);
        }
    }
    if let Primitive::Fusion { body, .. } = &node.primitive {
        for n in body.nodes() {
            sources_of(n, sources);
        }
    }
}
