//! Metal kernel launching: per-primitive encode dispatch, pipeline caching, argument packing,
//! and capturing launches so a plan can replay them against new buffer addresses.
use crate::codegen::dot_general::mps::plan_matmul;
use crate::codegen::dynamic_slice::mps::{MAX_RANK, copy};
use crate::codegen::utils::mps::Mps;
use crate::runtime::kernel::Launch as _;
use crate::runtime::kernel::mps::MpsLaunch;
use crate::runtime::wait::wait;
use crate::tensor::contiguous_strides;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, PoisonError};

use crate::graph::{Primitive, TensorType};
use crate::plan::Step;
use crate::stream::mps::{self, Completion};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar, Tensor};

unsafe extern "C" {
    /// Looks up kernel `name` and dispatches it once; returns nonzero on failure.
    fn lumen_mps_launch_kernel(
        name: *const c_char,
        buffers: *const *const u8,
        nbuffers: usize,
        args: *const *const u8,
        arg_lens: *const usize,
        nargs: usize,
        grid: *const usize,
        groups: i32,
        timed: i32,
        done: Completion,
        context: *mut c_void,
    ) -> i32;

    /// Returns the compute pipeline for kernel `name`, or null if no such kernel exists.
    fn lumen_mps_pipeline(name: *const c_char) -> *mut c_void;

    /// Dispatches an already-built pipeline; `groups` says whether `grid` counts threadgroups, `group` is the threadgroup size (0 for the default).
    pub(crate) fn lumen_mps_launch_pipeline(
        pipeline: *mut c_void,
        buffers: *const *const u8,
        nbuffers: usize,
        args: *const *const u8,
        arg_lens: *const usize,
        nargs: usize,
        grid: *const usize,
        groups: i32,
        group: usize,
        timed: i32,
        done: Completion,
        context: *mut c_void,
    ) -> i32;
}

/// A kernel launch recorded by `capture` instead of dispatched, with its raw buffer addresses.
pub(crate) struct Captured {
    pub(crate) buffers: Vec<*const u8>,
    pub(crate) args: Vec<Vec<u8>>,
    pub(crate) grid: Grid,
    pub(crate) kernel: String,
    pub(crate) label: &'static str,
}

/// Where a captured buffer pointer lives, as a byte offset into one of the plan's allocations.
#[derive(Debug, Clone, Copy)]
enum Slot {
    /// A null buffer pointer.
    Null,
    Workspace(usize),
    /// Operand index and byte offset into it.
    Operand(usize, usize),
    Output(usize),
}

/// A captured launch with its buffers rewritten as `Slot`s so it can run against other allocations.
#[derive(Debug, Clone)]
struct Replay {
    buffers: Vec<Slot>,
    args: Vec<Vec<u8>>,
    grid: Grid,
    kernel: String,
    label: &'static str,
}

/// A plan's captured launches, ready to replay.
#[derive(Debug, Clone)]
pub(crate) struct Prepared(Vec<Replay>);

/// Base address and byte length of each allocation a plan's kernels may touch.
pub(crate) struct Pointers<'a> {
    pub(crate) workspace: (*const u8, usize),
    pub(crate) operands: &'a [(*const u8, usize)],
    pub(crate) output: (*const u8, usize),
}

impl Pointers<'_> {
    /// Finds the allocation containing `p`; errors when none or more than one operand/output does.
    fn slot(&self, p: *const u8) -> Result<Slot, Unprepared> {
        let within = |(base, len): (*const u8, usize)| {
            let at = (p as usize).wrapping_sub(base as usize);
            (at < len.max(1)).then_some(at)
        };

        if p.is_null() {
            return Ok(Slot::Null);
        }

        if let Some(at) = within(self.workspace) {
            return Ok(Slot::Workspace(at));
        }

        let mut found = self
            .operands
            .iter()
            .enumerate()
            .filter_map(|(k, &operand)| within(operand).map(|at| Slot::Operand(k, at)))
            .chain(within(self.output).map(Slot::Output));

        match (found.next(), found.next()) {
            (Some(slot), None) => Ok(slot),
            (Some(_), Some(_)) => Err(Unprepared::Ambiguous),
            (None, _) => Err(Unprepared::Unplaced),
        }
    }

    /// Rebuilds a pointer from `slot` against these allocations.
    fn pointer(&self, slot: Slot) -> *const u8 {
        match slot {
            Slot::Null => std::ptr::null(),
            Slot::Workspace(at) => self.workspace.0.wrapping_add(at),
            Slot::Operand(k, at) => self.operands[k].0.wrapping_add(at),
            Slot::Output(at) => self.output.0.wrapping_add(at),
        }
    }
}

/// Why captured launches could not be turned into a `Prepared` replay.
pub(crate) enum Unprepared {
    /// A buffer pointer falls inside more than one operand or output.
    Ambiguous,
    /// A buffer pointer falls inside no known allocation.
    Unplaced,
}

/// Rewrites captured launches' buffer pointers as slots relative to the allocations in `at`.
pub(crate) fn prepare(captured: Vec<Captured>, at: &Pointers) -> Result<Prepared, Unprepared> {
    let launches = captured.into_iter().map(|c| {
        let buffers = c
            .buffers
            .iter()
            .map(|&p| at.slot(p))
            .collect::<Result<_, _>>()?;

        Ok(Replay {
            buffers,
            args: c.args,
            grid: c.grid,
            kernel: c.kernel,
            label: c.label,
        })
    });

    launches.collect::<Result<_, _>>().map(Prepared)
}

/// Dispatches prepared launches against the allocations in `at`; only the last launch holds `keep`.
pub(crate) fn replay(prepared: &Prepared, at: &Pointers, keep: Vec<Tensor>) -> Result<(), String> {
    let Some((last, first)) = prepared.0.split_last() else {
        return Ok(());
    };

    let mut buffers = Vec::new();

    for (r, keep) in first.iter().map(|r| (r, Vec::new())).chain([(last, keep)]) {
        buffers.clear();
        buffers.extend(r.buffers.iter().map(|&slot| at.pointer(slot)));

        Mps::launch(MpsLaunch {
            kernel: &r.kernel,
            buffers: &buffers,
            args: &r.args,
            grid: r.grid,
            keep,
            name: r.label,
        })?;
    }

    Ok(())
}

thread_local! {
    pub(crate) static CAPTURED: RefCell<Option<Vec<Captured>>> = const { RefCell::new(None) };
}

/// Runs `encode` with launches recorded rather than dispatched, returning what it launched.
pub(crate) fn capture(
    encode: impl FnOnce() -> Result<(), String>,
) -> Result<Vec<Captured>, String> {
    CAPTURED.with_borrow_mut(|c| *c = Some(Vec::new()));
    let encoded = encode();
    let captured = CAPTURED.with_borrow_mut(Option::take).unwrap_or_default();
    encoded.map(|()| captured)
}

/// Most argument buffers a kernel launch can carry.
pub(crate) const MAX_ARGS: usize = 31;

/// Returns the cached compute pipeline for `kernel`, building it on first use.
pub(crate) fn pipeline(kernel: &str) -> Result<*mut c_void, String> {
    static PIPELINES: Mutex<BTreeMap<String, usize>> = Mutex::new(BTreeMap::new());
    let mut pipelines = PIPELINES.lock().unwrap_or_else(PoisonError::into_inner);

    if let Some(&pipeline) = pipelines.get(kernel) {
        return Ok(pipeline as *mut c_void);
    }

    let name = CString::new(kernel).expect("kernel names have no NUL");
    let pipeline = unsafe { lumen_mps_pipeline(name.as_ptr()) };

    if pipeline.is_null() {
        return Err(format!(
            "the MPS kernel {kernel} could not run (no such kernel)"
        ));
    }

    pipelines.insert(kernel.to_owned(), pipeline as usize);
    Ok(pipeline)
}

/// A specialized encoder a step carries in its launch slot: `(step, inputs, output, scratch, keep)`.
pub(crate) type Encode = Box<
    dyn Fn(&Step, &[*const u8], *mut u8, *mut u8, Vec<Tensor>) -> Result<(), String> + Send + Sync,
>;

/// Encodes one plan step's kernels into the MPS stream, routing by primitive to its kernel family.
/// Errors when the output has more elements than 32-bit kernel indices reach.
pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    use Primitive::*;
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
                    let (outer, inner) = (&out.shape[..*dimension], &out.shape[dimension + 1..]);

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
        TakeAlongAxis { .. } => unreachable!("rewritten as a gather (passes/take_along_axis.rs)"),
    }
}

/// Scratch bytes a step's kernels need beyond its output, for primitives that use any.
pub(crate) fn scratch_bytes(p: &Primitive, inputs: &[&TensorType], _output: &TensorType) -> usize {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
            let accum = crate::codegen::reduction::mps::accum_dtype(p, inputs[0].dtype);
            crate::codegen::reduction::mps::scratch_bytes(inputs[0], axes, accum)
        }
        Primitive::Cumsum {
            axis, accum_dtype, ..
        } => crate::codegen::scan::mps::scratch_bytes(inputs[0], *axis, *accum_dtype),
        Primitive::Fusion { body, .. } => crate::graph_compiler::mps::fusion_scratch_bytes(body),
        Primitive::LinearCrossEntropy => {
            crate::codegen::dot_general::mps::linear_cross_entropy_scratch(inputs[0], inputs[1])
        }
        Primitive::ScatterAdd { axis } => {
            crate::codegen::scatter_add::mps::scratch_bytes(inputs[0], inputs[1], *axis)
        }
        _ => 0,
    }
}

/// Packs a `u32` kernel argument.
pub(crate) fn u32_arg(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

/// Packs a `usize` as a `u64` kernel argument.
pub(crate) fn u64_arg(v: usize) -> Vec<u8> {
    (v as u64).to_ne_bytes().to_vec()
}

/// Packs dimensions as a `u64` array argument; empty becomes `[0]` so the buffer is never zero-sized.
pub(crate) fn dims_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u64> = v.into_iter().map(|d| d as u64).collect();

    if v.is_empty() {
        v.push(0);
    }

    as_bytes(&v).to_vec()
}

/// Packs dimensions as a `u32` array argument; empty becomes `[0]` so the buffer is never zero-sized.
pub(crate) fn dims32_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u32> = v.into_iter().map(|d| d as u32).collect();

    if v.is_empty() {
        v.push(0);
    }

    as_bytes(&v).to_vec()
}

/// Packs `value` converted to `dtype` as a single-element argument.
pub(crate) fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

/// Bytes each elementwise thread handles; matches `BYTES_PER_THREAD` in `kernels.metal`.
pub(crate) const BYTES_PER_THREAD: usize = 16;
/// Thread count above which elementwise grids start giving threads several elements.
const GRID_THREADS: usize = 32 << 10;

/// One-dimensional grid for an elementwise kernel over `n` elements, giving each thread up to 16 bytes once `n` is large.
pub(crate) fn elementwise_grid(n: usize, dtype: DType) -> Grid {
    let most = (BYTES_PER_THREAD / dtype.size_of()).max(1);
    Grid::Threads([n.div_ceil((n / GRID_THREADS).clamp(1, most)), 1, 1])
}

pub(crate) use crate::custom_op::mps::Grid;

/// Dispatches an empty `event_marker` kernel whose completion signals `recorded`, fenced on both sides.
pub(crate) fn mark_event(
    recorded: std::sync::Arc<crate::stream::event::Recorded>,
) -> Result<(), String> {
    let c_name = CString::new("event_marker").expect("kernel names have no NUL");
    let (context, done) = mps::submit_event(recorded);
    let sizes = [1usize, 1, 1];
    mps::fence();

    let status = unsafe {
        lumen_mps_launch_kernel(
            c_name.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
            0,
            sizes.as_ptr(),
            0,
            1,
            done,
            context,
        )
    };

    if status != 0 {
        mps::cancel(context);
        return Err(format!("recording an MPS event failed ({status})"));
    }

    mps::fence();
    Ok(())
}
