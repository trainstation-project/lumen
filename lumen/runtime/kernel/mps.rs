//! Launching Metal kernels: each dispatched on the MPS stream through its cached pipeline, or recorded inside
//! `capture` so a plan can replay it against new buffer addresses; the argument packing and grids kernels launch
//! with; and the event markers streams record. Also compiling generated Metal sources into the shared library, and
//! the encoder that launches each fused MPS step with the right grid.

use crate::codegen::attention::AttentionCodegen;
use crate::codegen::attention::mps::decodes;
use crate::codegen::dot_general::mps as gemm;
use crate::codegen::dot_general::mps::GemmConfig;
use crate::codegen::elementwise::mps::Tuning;
use crate::codegen::kinds::{gemm_dot, reduction_root, row_reductions};
use crate::codegen::layout::transpose_tiling;
use crate::codegen::reduction::mps::{has_epilogue, split_launches};
use crate::codegen::row::mps::row_groups;
use crate::codegen::store::mps::strided_load;
use crate::codegen::utils::mps::Mps;
use crate::codegen::utils::mps::loads_strided;
pub(crate) use crate::custom_op::mps::Grid;
use crate::graph::{Graph, Primitive, TensorType, Var, View};
use crate::graph_compiler::passes::apple_neural_engine::mps as ane;
use crate::graph_compiler::passes::attention;
use crate::graph_compiler::passes::horizontal::segments;
use crate::plan::{Buffer, StepBase};
use crate::runtime::kernel::Launch as _;
use crate::runtime::stream::mps::{self, Completion};
use crate::tensor::contiguous_strides;
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar, Tensor};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{CString, c_char, c_void};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Mutex, OnceLock, PoisonError};

unsafe extern "C" {
    /// Compiles `source` and adds its kernels to the shared library; returns nonzero on failure.
    fn lumen_mps_compile_kernels(source: *const c_char) -> i32;

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

/// Dispatches an empty `event_marker` kernel whose completion signals `recorded`, fenced on both sides.
pub(crate) fn mark_event(
    recorded: std::sync::Arc<crate::runtime::event::Recorded>,
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

/// One Metal launch: a kernel by name, its buffers and by-value arguments, its grid, the tensors its command
/// buffer keeps alive, and the step label it is profiled under.
pub(crate) struct MpsLaunch<'a> {
    pub(crate) kernel: &'a str,
    pub(crate) buffers: &'a [*const u8],
    pub(crate) args: &'a [Vec<u8>],
    pub(crate) grid: Grid,
    pub(crate) keep: Vec<Tensor>,
    pub(crate) name: &'static str,
}

/// MPS dispatches a kernel's pipeline on the MPS stream, or records it inside `capture`.
impl super::Launch for Mps {
    type Args<'a> = MpsLaunch<'a>;

    /// Launches `kernel`, or records it when inside `capture`; a grid with a zero dimension is skipped.
    fn launch(args: MpsLaunch<'_>) -> Result<(), String> {
        let MpsLaunch {
            kernel,
            buffers,
            args,
            grid,
            keep,
            name,
        } = args;

        let (sizes, groups, group) = match grid {
            Grid::Threads(sizes) => (sizes, false, 0),
            Grid::Groups(sizes) => (sizes, true, 0),
            Grid::GroupsOf { groups, threads } => (groups, true, threads),
            Grid::Elementwise { threads, group } => ([threads, 1, 1], false, group),
        };

        if sizes.contains(&0) {
            crate::profiler::clear_next_launch();
            return Ok(());
        }

        let pipeline = pipeline(kernel)?;

        if args.len() > MAX_ARGS {
            return Err(format!(
                "the MPS kernel {kernel} takes at most {MAX_ARGS} arguments"
            ));
        }

        let captured = CAPTURED.with_borrow_mut(|c| {
            c.as_mut().map(|c| {
                c.push(Captured {
                    buffers: buffers.to_vec(),
                    args: args.to_vec(),
                    grid,
                    kernel: kernel.to_owned(),
                    label: name,
                })
            })
        });

        if captured.is_some() {
            return Ok(());
        }

        let mut pointers = [std::ptr::null::<u8>(); MAX_ARGS];
        let mut lens = [0usize; MAX_ARGS];

        for (k, a) in args.iter().enumerate() {
            pointers[k] = a.as_ptr();
            lens[k] = a.len();
        }

        let (context, done, timed) = mps::submit(keep, name, kernel);

        let status = unsafe {
            lumen_mps_launch_pipeline(
                pipeline,
                buffers.as_ptr(),
                buffers.len(),
                pointers.as_ptr(),
                lens.as_ptr(),
                args.len(),
                sizes.as_ptr(),
                groups.into(),
                group,
                timed.into(),
                done,
                context,
            )
        };

        if status != 0 {
            mps::cancel(context);
            return Err(format!("the MPS kernel {kernel} could not run ({status})"));
        }

        Ok(())
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

/// Kernel templates and helpers every generated Metal source is compiled against.
const PRELUDE: &str = concat!(
    include_str!("../../codegen/utils/mps/prelude.metal"),
    include_str!("../../codegen/random/mps/philox.metal"),
    indoc::indoc!(
        "

        #define TEMPLATES_ONLY
        "
    ),
    include_str!("../../codegen/elementwise/mps/kernel.metal"),
    include_str!("../../codegen/reduction/mps/kernel.metal"),
    include_str!("../../codegen/dot_general/mps/integer/matmul_kernel.metal"),
    include_str!("../../codegen/dot_general/mps/integer/matmul_wide_kernel.metal"),
    include_str!("../../codegen/dot_general/mps/tiled/matmul_sg_kernel.metal"),
    include_str!("../../codegen/dot_general/mps/skinny/kernel.metal"),
    include_str!("../../codegen/dot_general/mps/gemv/kernel.metal"),
    include_str!("../../codegen/dot_general/mps/linear_cross_entropy/kernel.metal"),
    include_str!("../../codegen/attention/mps/forward_kernel.metal"),
    include_str!("../../codegen/attention/mps/backward_kernel.metal"),
);

/// Metal Performance Primitives matmul and attention templates, compiled only into MPP kernels.
const MPP: &str = concat!(
    include_str!("../../codegen/dot_general/mps/mpp/kernel.metal"),
    include_str!("../../codegen/attention/mps/forward_mpp_kernel.metal"),
);

/// Compiles `sources`, attention kernels on Metal Performance Primitives against their templates and the rest
/// against the standard prelude.
pub(crate) fn compile_routed(sources: BTreeMap<String, String>) -> Result<(), String> {
    let (mpp, sources): (BTreeMap<_, _>, BTreeMap<_, _>) = sources
        .into_iter()
        .partition(|(_, source)| source.contains("flash_attention_mpp"));

    compile_kernels(sources)?;
    compile_mpp_kernels(mpp)
}

/// Compiles `sources` against the standard prelude.
pub(crate) fn compile_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(PRELUDE, sources)
}

/// Compiles `sources` against the prelude plus the MPP templates.
pub(crate) fn compile_mpp_kernels(sources: BTreeMap<String, String>) -> Result<(), String> {
    compile_with(&format!("{PRELUDE}\n{MPP}"), sources)
}

/// Compiles `sources` as one Metal library after `prelude` and records them as compiled.
fn compile_with(prelude: &str, sources: BTreeMap<String, String>) -> Result<(), String> {
    if sources.is_empty() {
        return Ok(());
    }

    let source: String = std::iter::once(prelude)
        .chain(sources.values().map(String::as_str))
        .collect::<Vec<_>>()
        .join("\n");

    let source = CString::new(source).expect("Metal source has no NUL");
    let status = unsafe { lumen_mps_compile_kernels(source.as_ptr()) };

    if status != 0 {
        return Err(format!("compiling the fused MPS kernels failed ({status})"));
    }

    let mut compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    compiled.extend(sources);
    Ok(())
}

/// Every compiled kernel source by name.
pub(crate) static SOURCES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// Takes `sources`, keeping only kernels not already compiled.
pub(crate) fn uncompiled(sources: &mut BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut sources = std::mem::take(sources);
    let compiled = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.retain(|name, _| !compiled.contains_key(name));
    sources
}

/// Names of `step`'s kernel variants for `configs`: each generated by `generate` (its name and source) on its first
/// use and kept in the step's memo, the new ones compiled together by `compile` (skipping kernels already compiled).
pub(crate) fn variants<C: std::fmt::Debug>(
    step: &StepBase,
    configs: &[C],
    generate: impl Fn(&C) -> (String, String),
    compile: impl FnOnce(BTreeMap<String, String>) -> Result<(), String>,
) -> Result<Vec<String>, String> {
    let mut pending = BTreeMap::new();

    let names = configs
        .iter()
        .map(|config| {
            let mut hasher = DefaultHasher::new();
            (std::any::type_name::<C>(), format!("{config:?}")).hash(&mut hasher);

            let name = step.memo.get(hasher.finish(), || {
                let (name, source) = generate(config);
                pending.insert(name.clone(), source);
                Ok(name)
            })?;

            Ok(name.as_ref().clone())
        })
        .collect::<Result<Vec<String>, String>>()?;

    compile(uncompiled(&mut pending))?;
    Ok(names)
}

/// Whether this Metal toolchain supports Metal Performance Primitives, probed once: a kernel defined only
/// under `LUMEN_MPP`, compiling (and logging) no error either way.
pub(crate) fn mpp_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();

    *AVAILABLE.get_or_init(|| {
        let probe = "#ifdef LUMEN_MPP\nkernel void mpp_probe(device int *out [[buffer(0)]]) { out[0] = 1; }\n#endif\n";
        let sources = BTreeMap::from([("mpp_probe".to_owned(), probe.to_owned())]);
        let name = CString::new("mpp_probe").expect("no NUL");

        compile_mpp_kernels(sources).is_ok() && !unsafe { lumen_mps_pipeline(name.as_ptr()) }.is_null()
    })
}

/// How a fused kernel is launched.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Launch {
    /// Flash attention forward.
    Attention,
    /// Flash attention backward.
    AttentionBackward,
    /// GEMM with a fused epilogue.
    Gemm,
    /// Row-wise reductions in one threadgroup per row group.
    Rows,
    /// Standalone reduction kernel.
    Reduction,
    /// Tiled transpose.
    Transpose,
    /// Plain elementwise fusion.
    Elementwise,
    /// A chain of constant-start updates written into its operand.
    Updates,
}

/// Launch kind and tuning of each compiled kernel by name.
static LAUNCHES: Mutex<BTreeMap<String, (Launch, Option<Tuning>)>> = Mutex::new(BTreeMap::new());

/// Classifies a fusion body by the launch its kernel needs.
pub(crate) fn launch_of(body: &Graph) -> Launch {
    let update = |n: &crate::graph::Node| {
        matches!(
            n.primitive,
            Primitive::DynamicUpdateSlice | Primitive::Full { .. }
        )
    };

    let updating = body
        .nodes()
        .iter()
        .any(|n| matches!(n.primitive, Primitive::DynamicUpdateSlice));

    if updating && body.nodes().iter().all(update) {
        Launch::Updates
    } else if attention::of_body(body).is_some() {
        Launch::Attention
    } else if attention::backward_of_body(body).is_some() {
        Launch::AttentionBackward
    } else if gemm_dot(body).is_some() {
        Launch::Gemm
    } else if !row_reductions(body).is_empty() {
        Launch::Rows
    } else if reduction_root(body).is_some() {
        Launch::Reduction
    } else if transpose_tiling(body).is_some() {
        Launch::Transpose
    } else {
        Launch::Elementwise
    }
}

/// Records how kernel `name` for `body` launches under `tuning`.
pub(crate) fn record_launch(name: &str, body: &Graph, tuning: Option<Tuning>) {
    let launch = (launch_of(body), tuning);
    let mut launches = LAUNCHES.lock().unwrap_or_else(PoisonError::into_inner);
    launches.insert(name.to_owned(), launch);
}

/// Launch kind and tuning of kernel `name`, deriving and caching them from `body` if unrecorded.
fn launch_kind(name: &str, body: &Graph) -> (Launch, Option<Tuning>) {
    let mut launches = LAUNCHES.lock().unwrap_or_else(PoisonError::into_inner);

    if let Some(&launch) = launches.get(name) {
        return launch;
    }

    let launch = launch_of(body);
    let tuning = crate::codegen::utils::mps::tuning_of(body);
    launches.insert(name.to_owned(), (launch, tuning));
    (launch, tuning)
}

/// Generated Metal source of fusion kernel `name`, if compiled.
#[cfg(feature = "python")]
pub(crate) fn fusion_source(name: &str) -> Option<String> {
    let sources = SOURCES.lock().unwrap_or_else(PoisonError::into_inner);
    sources.get(name).cloned()
}

/// Strided-load kernel variants by base kernel name and input strides.
static VARIANTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

/// Type of `t` with its dtype swapped for the same-width unsigned integer, so copies move bits.
fn carried(t: &Tensor) -> TensorType {
    let carrier = match t.dtype().size_of() {
        1 => DType::U8,
        2 => DType::U16,
        4 => DType::U32,
        _ => DType::U64,
    };

    TensorType::new(carrier, t.shape())
}

/// Copies `src` into `dst` bit for bit with an elementwise kernel, honoring both tensors' strides.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let mut body = Graph::new();
    let x = body.input(carried(dst));
    body.set_outputs(&[x]).expect("the copied value");
    run_loop(body, "copy_", dst, src, false)
}

/// Fills `dst` with `value` by broadcasting its bits from a by-value scalar.
pub(crate) fn fill(dst: &Tensor, value: Scalar) -> Result<(), String> {
    let ty = carried(dst);
    let mut body = Graph::new();
    let x = body.input(TensorType::new(ty.dtype, &[]));

    let broadcast = Primitive::BroadcastInDim {
        shape: ty.shape.clone(),
        broadcast_dimensions: Vec::new(),
    };

    let v = body.apply(broadcast, &[x])?;
    body.set_outputs(&[v]).expect("the filled value");
    let bits = crate::codegen::utils::triton::bits(dst.dtype(), value);

    let scalar = match ty.dtype {
        DType::U8 => Tensor::from_slice(&[bits as u8], ty.dtype),
        DType::U16 => Tensor::from_slice(&[bits as u16], ty.dtype),
        DType::U32 => Tensor::from_slice(&[bits as u32], ty.dtype),
        _ => Tensor::from_slice(&[bits], ty.dtype),
    };

    run_loop(body, "fill_", dst, &scalar, true)
}

/// Compiles (once) and runs the one-input elementwise `body` from `src` into `dst`.
/// `by_value` passes `src` as a scalar argument instead of a buffer.
fn run_loop(
    body: Graph,
    label: &'static str,
    dst: &Tensor,
    src: &Tensor,
    by_value: bool,
) -> Result<(), String> {
    let views = match dst.is_contiguous() {
        true => HashMap::new(),
        false => HashMap::from([(0, (0, dst.strides().to_vec()))]),
    };

    let config = crate::graph_compiler::config::config();
    let (name, source) =
        crate::codegen::utils::mps::kernel(&body, &[by_value], &config, &views, None)?;

    let compiled = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains_key(&name);

    if !compiled {
        compile_kernels(BTreeMap::from([(name.clone(), source)]))?;
        record_launch(&name, &body, crate::codegen::utils::mps::tuning_of(&body));
    }

    let strided = !by_value && !src.is_contiguous();
    let input = body.type_of(body.inputs()[0]).clone();
    let output = body.type_of(body.outputs()[0]).clone();

    let step = StepBase {
        primitive: Primitive::Fusion { name, label, body },
        label,
        inputs: vec![(
            match by_value {
                true => Buffer::Scalar(0),
                false => Buffer::Input(0),
            },
            input,
        )],
        views: vec![strided.then(|| View {
            offset: 0,
            strides: src.strides().to_vec(),
        })],
        output: (Buffer::Output(0), output),
        extra_outputs: Vec::new(),
        output_views: vec![None],
        scratch: None,
        scope: &[],
        sources: Vec::new(),
        tuned: None,
        memo: Default::default(),
    };

    let keep = vec![src.holding(), dst.holding()];
    let inputs = [src.data_ptr().cast_const()];
    encode(&step, &inputs, dst.data_ptr(), std::ptr::null_mut(), keep)
}

/// Name of a kernel variant of `name` that loads the step's strided input views directly, compiling it on
/// first use; `None` when no view needs it.
fn strided_variant(step: &StepBase, name: &str, body: &Graph) -> Result<Option<String>, String> {
    if step.views.iter().all(Option::is_none) {
        return Ok(None);
    }

    let (launch, _) = launch_kind(name, body);

    if matches!(launch, Launch::Attention | Launch::AttentionBackward) {
        return attention_variant(step, name, body, launch).map(Some);
    }

    let strided: Vec<(usize, &[usize])> = step
        .views
        .iter()
        .enumerate()
        .filter_map(|(k, view)| view.as_ref().map(|view| (k, view.strides.as_slice())))
        .filter(|&(k, _)| loads_strided(body, k))
        .collect();

    if strided.is_empty() {
        return Ok(None);
    }

    let key = format!("{name}{strided:?}");

    if let Some(variant) = VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(Some(variant.clone()));
    }

    let Some(mut source) = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(name)
        .cloned()
    else {
        return Ok(None);
    };

    for &(k, strides) in &strided {
        let shape = &step.inputs[k].1.shape;

        if let Some(rewritten) = strided_load(&source, name, k, shape, strides) {
            source = rewritten;
        }
    }

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let variant = format!("fusion_{:016x}", hasher.finish());
    let source = source.replace(name, &variant);
    compile_kernels(BTreeMap::from([(variant.clone(), source)]))?;
    let launch = launch_kind(name, body);

    LAUNCHES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(variant.clone(), launch);

    VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variant.clone());

    Ok(Some(variant))
}

/// The strides attention step `step` (of fused `body`) reads its operands and writes its outputs at.
pub(crate) fn attention_strides(step: &StepBase, body: &Graph) -> attention::Strides {
    let outputs: Vec<(usize, Vec<usize>)> = std::iter::once(&step.output)
        .chain(&step.extra_outputs)
        .zip(&step.output_views)
        .map(|((_, ty), view)| match view {
            Some(view) => (view.offset, view.strides.clone()),
            None => (0, contiguous_strides(&ty.shape)),
        })
        .collect();

    attention::Strides::of(body, &step.input_strides(), &outputs)
}

/// Name of attention kernel `name` generated for the step's operand layouts (any strides but a
/// contiguous last dim), compiling it on first use.
fn attention_variant(
    step: &StepBase,
    name: &str,
    body: &Graph,
    launch: Launch,
) -> Result<String, String> {
    let strides = attention_strides(step, body);
    let mut layout: Vec<_> = strides.inputs.iter().collect();
    layout.sort();
    let mut views: Vec<_> = strides.outputs.iter().collect();
    views.sort();
    let key = format!("{name}{layout:?}{views:?}");

    if let Some(variant) = VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(variant.clone());
    }

    let unmatched = || {
        format!(
            "{}: no attention kernel reads its operands' layout",
            step.label
        )
    };

    let (variant, source) = match launch {
        Launch::Attention => {
            let a = attention::of_body_strided(body, &strides).ok_or_else(unmatched)?;
            Mps::forward(body, &a, &strides)?
        }
        _ => {
            let b = attention::backward_of_body_strided(body, &strides).ok_or_else(unmatched)?;
            Mps::backward(body, &b, &strides)?
        }
    };

    let compiled = SOURCES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .contains_key(&variant);

    if !compiled {
        compile_routed(BTreeMap::from([(variant.clone(), source)]))?;
    }

    let launch = launch_kind(name, body);

    LAUNCHES
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(variant.clone(), launch);

    VARIANTS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(key, variant.clone());

    Ok(variant)
}

/// Encodes fused step `step` reading `inputs` into `output`, using `scratch` and keeping `keep` alive.
pub(crate) fn encode(
    step: &StepBase,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { name, .. } = &step.primitive else {
        unreachable!("a fusion")
    };

    encode_with(step, name, None, None, inputs, output, scratch, keep)
}

/// Encodes a fused step, launching the tuned GEMM variant `gemm` or attention variant `attention_tuned`
/// (its kernel and tuning) when given. `inputs` holds the body inputs followed by extra output pointers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_with(
    step: &StepBase,
    name: &str,
    gemm: Option<(&str, GemmConfig)>,
    attention_tuned: Option<(&str, crate::codegen::attention::mps::Tuning)>,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let Primitive::Fusion { body, .. } = &step.primitive else {
        unreachable!("a fusion")
    };

    let variant = strided_variant(step, name, body)?;
    let name = variant.as_deref().unwrap_or(name);

    if name.starts_with(crate::graph::NEURAL_ENGINE) {
        return ane::encode(step, inputs, output, keep);
    }

    let (inputs, extra) = inputs.split_at(body.inputs().len());
    // By-value inputs become kernel arguments; the rest are bound as buffers.
    let (mut device, mut scalars) = (Vec::new(), Vec::new());

    for (&p, (b, ty)) in inputs.iter().zip(&step.inputs) {
        match b {
            Buffer::Scalar(_) => {
                scalars.push(unsafe { std::slice::from_raw_parts(p, ty.dtype.size_of()) }.to_vec())
            }
            _ => device.push(p),
        }
    }

    let inputs = device.as_slice();
    let (kind, tuning) = launch_kind(name, body);

    // Attention buffers: inputs, outputs, then the dropout random state, which must be bound as buffers.
    let attention_buffers = |dropout: bool| -> Result<Vec<*const u8>, String> {
        let state = match (dropout, scalars.len()) {
            (false, _) | (true, 2) => 0,
            (true, 0) => 2,
            _ => return Err(format!("{}: its random state is half by value", step.label)),
        };
        let (ins, state) = inputs.split_at(inputs.len() - state);
        let mut buffers = ins.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        buffers.extend(state);
        Ok(buffers)
    };

    if kind == Launch::Updates {
        // An out-of-place chain starts from a copy of its operand; each thread writes one element of
        // every update.
        if output.cast_const() != inputs[0] {
            let x = &step.inputs[0].1;

            crate::codegen::dynamic_slice::mps::copy(
                inputs[0],
                output,
                x,
                keep.clone(),
                step.label,
            )?;
        }

        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        let grid = Grid::Threads([step.inputs[1].1.numel(), 1, 1]);
        return Mps::launch(MpsLaunch {
            kernel: name,
            buffers: &buffers,
            args: &scalars,
            grid,
            keep,
            name: step.label,
        });
    }

    if kind == Launch::Attention {
        let a = attention::of_body(body).expect("an attention");
        let (outer, heads) = attention::grid_batch(&a.batch);

        // Decode attention runs one threadgroup per query, otherwise one of its tuning's simdgroups per
        // its rows of them.
        let t =
            attention_tuned.map_or_else(|| crate::codegen::attention::mps::tuning(&a), |(_, t)| t);
        let (groups, rows) = (t.groups, t.rows());

        let grid = match decodes(&a) {
            true => Grid::Groups([a.sq, heads, outer]),
            false => Grid::GroupsOf {
                groups: [a.sq.div_ceil(rows * groups), heads, outer],
                threads: 32 * groups,
            },
        };

        let name = attention_tuned.map_or(name, |(kernel, _)| kernel);
        let buffers = attention_buffers(a.dropout.is_some())?;
        return Mps::launch(MpsLaunch {
            kernel: name,
            buffers: &buffers,
            args: &scalars,
            grid,
            keep,
            name: step.label,
        });
    }

    if kind == Launch::AttentionBackward {
        let b = attention::backward_of_body(body).expect("an attention backward");
        let (outer, heads) = attention::grid_batch(&b.batch);
        let rows = if b.dv_out.is_some() { b.sk } else { b.sq };
        let buffers = attention_buffers(b.dropout.is_some())?;

        // The fused dK/dV kernel accumulates dQ atomically, so dQ starts at zero.
        if b.dv_out.is_some() && b.dq_out.is_some() {
            let n = body.type_of(body.outputs()[2]).numel();

            let args = [
                element_arg(DType::F32, Scalar::Float(0.0)),
                u32_arg(n as u32),
            ];

            let grid = elementwise_grid(n, DType::F32);
            Mps::launch(MpsLaunch {
                kernel: "fill_4",
                buffers: &[extra[1]],
                args: &args,
                grid,
                keep: Vec::new(),
                name: "full (dQ)",
            })?;
        }

        let grid = Grid::Groups([rows.div_ceil(64), heads, outer]);
        return Mps::launch(MpsLaunch {
            kernel: name,
            buffers: &buffers,
            args: &scalars,
            grid,
            keep,
            name: step.label,
        });
    }

    if kind == Launch::Gemm {
        let dot = gemm_dot(body).expect("a matmul");
        let ty = |v: Var| body.type_of(v);
        let (lhs, rhs) = (ty(dot.inputs[0]), ty(dot.inputs[1]));

        let strides = |v: Var| {
            let k = body.inputs().iter().position(|&i| i == v);

            k.and_then(|k| step.views[k].as_ref()).map_or_else(
                || contiguous_strides(&ty(v).shape),
                |view| view.strides.clone(),
            )
        };

        let (ls, rs) = (strides(dot.inputs[0]), strides(dot.inputs[1]));
        let tuned = gemm;

        let config = tuned
            .as_ref()
            .map(|&(_, c)| c)
            .or_else(|| gemm::gemv_config(body));

        let plan = crate::codegen::dot_general::mps::plan_matmul(
            &dot.primitive,
            lhs,
            rhs,
            ty(dot.output),
            &ls,
            &rs,
            step.label,
            config.map(|c| (c.bm, c.bn)),
        )?;

        let kernel = match (&tuned, config, plan.small, plan.mid) {
            (Some((tuned, _)), ..) => tuned.to_string(),
            (None, Some(_), ..) => name.to_owned(),
            (None, None, true, _) => format!("{name}_small"),
            (None, None, _, true) => format!("{name}_mid"),
            _ => name.to_owned(),
        };

        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        let args: Vec<Vec<u8>> = scalars.iter().cloned().chain([dims_arg(plan.p)]).collect();

        // Row-tile epilogues run as two launches: the GEMM writes tiles to scratch, then a row kernel finishes.
        if let Some((offsets, _)) = gemm::tile_scratch(body) {
            let at = |o: usize| scratch.wrapping_add(o).cast_const();

            // The dot's value, then each tile reduction's partials and a cut kept in scratch, bound in
            // that order after both kernels' outputs.
            let (dotv, tiles): (*const u8, Vec<*const u8>) = (
                at(offsets[0]),
                offsets[1..].iter().map(|&o| at(o)).collect(),
            );

            let [m, n] = [plan.p[0], plan.p[1]];
            let mut first = buffers.clone();
            first.push(dotv);
            first.extend(&tiles);
            let name = tuned.as_ref().map_or(name, |(tuned, _)| tuned);

            let grid = match config {
                Some(c) if c.algorithm != crate::codegen::dot_general::mps::Algorithm::Tiled => {
                    Grid::GroupsOf {
                        groups: [n.div_ceil(c.bn), m.div_ceil(c.bm), 1],
                        threads: c.threads,
                    }
                }
                _ => Grid::Groups([n.div_ceil(gemm::ROW_TILE), m.div_ceil(gemm::ROW_TILE), 1]),
            };

            // The profiler sees each pass's own reads and writes: the partials pass writes the dot's
            // value and the tiles' partials, which the combine reads back.
            let passes = gemm::tile_passes(body).expect("a multi-tile row matmul");
            let strides = step.input_strides();

            // The step's outputs the partials pass writes (`early`), or the combine the rest.
            let written = |early: bool| -> (Vec<TensorType>, Vec<Vec<usize>>) {
                let outputs = std::iter::once(&step.output).chain(&step.extra_outputs);

                outputs
                    .zip(step.output_strides())
                    .enumerate()
                    .filter(|(k, _)| passes.early.contains(k) == early)
                    .map(|(_, ((_, ty), strides))| (ty.clone(), strides))
                    .unzip()
            };

            let io = |reads: &[usize], scratch: &[TensorType]| {
                let types = reads.iter().map(|&i| step.inputs[i].1.clone());
                let read_strides = reads.iter().map(|&i| strides[i].clone());

                (
                    types.chain(scratch.iter().cloned()).collect(),
                    read_strides
                        .chain(scratch.iter().map(|t| contiguous_strides(&t.shape)))
                        .collect(),
                )
            };

            crate::profiler::next_launch(
                crate::graph::intern(format!("{} (partials)", step.label)),
                || {
                    let (inputs, input_strides) = io(&passes.partials, &[]);
                    let (outputs, output_strides) = written(true);
                    let scratch = passes.scratch.iter();

                    crate::profiler::IoTypes {
                        inputs,
                        outputs: outputs
                            .into_iter()
                            .chain(scratch.clone().cloned())
                            .collect(),
                        input_strides,
                        output_strides: output_strides
                            .into_iter()
                            .chain(scratch.map(|t| contiguous_strides(&t.shape)))
                            .collect(),
                    }
                },
            );

            Mps::launch(MpsLaunch {
                kernel: &format!("{name}_partials"),
                buffers: &first,
                args: &args,
                grid,
                keep: keep.clone(),
                name: step.label,
            })?;

            crate::profiler::next_launch(
                crate::graph::intern(format!("{} (combine)", step.label)),
                || {
                    let (mut inputs, mut input_strides) = io(&passes.combine, &passes.scratch);
                    let (outputs, output_strides) = written(false);

                    // A cut that is an output, written by the partials pass, read back here.
                    if let Some(k) = passes.read {
                        let all = std::iter::once(&step.output).chain(&step.extra_outputs);
                        let (_, ty) = all.clone().nth(k).expect("the cut output");
                        inputs.push(ty.clone());
                        input_strides.push(step.output_strides()[k].clone());
                    }

                    crate::profiler::IoTypes {
                        inputs,
                        outputs,
                        input_strides,
                        output_strides,
                    }
                },
            );

            let mut rows = inputs.to_vec();
            rows.push(dotv);
            rows.push(output.cast_const());
            rows.extend(extra);
            rows.extend(&tiles);
            let grid = Grid::Groups([row_groups(body), 1, 1]);
            let combine = format!("{name}_combine");
            return Mps::launch(MpsLaunch {
                kernel: &combine,
                buffers: &rows,
                args: &scalars,
                grid,
                keep,
                name: step.label,
            });
        }

        let (kernel, grid) = match gemm::row_dot(body) {
            Some(_) => (
                name.to_owned(),
                Grid::Groups([1, plan.p[0].div_ceil(gemm::single_rows(plan.p[0]).0), 1]),
            ),
            None => match (config, plan.grid) {
                (Some(c), Grid::Groups(groups)) => (
                    kernel,
                    Grid::GroupsOf {
                        groups,
                        threads: c.threads,
                    },
                ),
                (_, grid) => (kernel, grid),
            },
        };

        return Mps::launch(MpsLaunch {
            kernel: &kernel,
            buffers: &buffers,
            args: &args,
            grid,
            keep,
            name: step.label,
        });
    }

    if kind == Launch::Rows {
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);

        return Mps::launch(MpsLaunch {
            kernel: name,
            buffers: &buffers,
            args: &scalars,
            grid: Grid::Groups([row_groups(body), 1, 1]),
            keep,
            name: step.label,
        });
    }

    if kind == Launch::Reduction {
        let root = reduction_root(body).expect("a reduction");
        let x = body.type_of(root.inputs[0]);
        let label = step.label;
        let last = has_epilogue(body).then(|| format!("{name}_final"));

        return crate::codegen::reduction::mps::encode_reduction(
            step,
            &root.primitive,
            x,
            Some(name),
            last.as_deref(),
            split_launches(body),
            label,
            inputs,
            output,
            extra,
            &scalars,
            scratch,
            keep,
        );
    }

    if kind == Launch::Transpose {
        let tiling = transpose_tiling(body).expect("a tiled transpose");
        let mut buffers = inputs.to_vec();
        buffers.push(output.cast_const());
        buffers.extend(extra);
        return Mps::launch(MpsLaunch {
            kernel: name,
            buffers: &buffers,
            args: &scalars,
            grid: tiling.grid(),
            keep,
            name: step.label,
        });
    }

    let out = &step.output.1;
    let n = out.numel();

    let Some(Tuning::Elementwise {
        per_thread, group, ..
    }) = tuning
    else {
        unreachable!("an elementwise fusion's tuning")
    };

    // A horizontal fusion of several element counts runs each on its own range of threads.
    let threads = segments(body)
        .iter()
        .map(|(numel, _)| numel.div_ceil(per_thread))
        .sum();

    let grid = Grid::Elementwise { threads, group };

    let buffers: Vec<*const u8> = inputs
        .iter()
        .copied()
        .chain([output.cast_const()])
        .chain(extra.iter().copied())
        .collect();

    let args: Vec<Vec<u8>> = scalars.into_iter().chain([u32_arg(n as u32)]).collect();
    Mps::launch(MpsLaunch {
        kernel: name,
        buffers: &buffers,
        args: &args,
        grid,
        keep,
        name: step.label,
    })
}

/// Scratch bytes a fusion kernel needs: tile buffers for row-tile GEMMs or partials for reductions.
pub(crate) fn fusion_scratch_bytes(body: &Graph) -> usize {
    if let Some((_, bytes)) = gemm::tile_scratch(body) {
        return bytes;
    }

    match reduction_root(body).filter(|r| {
        matches!(
            r.primitive,
            Primitive::ReduceSum { .. } | Primitive::ReduceMax { .. }
        )
    }) {
        Some(root) => {
            let (Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes }) =
                &root.primitive
            else {
                unreachable!("a reduction")
            };

            let x = body.type_of(root.inputs[0]);
            let accum = body.type_of(root.output).dtype;
            crate::codegen::reduction::mps::scratch_bytes(x, axes, accum)
        }
        None => 0,
    }
}

/// The Metal kernel of an untuned fusion (elementwise, row, reduction, transpose, updates, attention backward, a GEMM the tiled kernels don't take, or the Neural Engine), launched by its kind.
pub(crate) struct Fusion;

impl crate::codegen::kernel::Kernel for Fusion {
    type Step = crate::plan::MpsStep;
    type Buffers = crate::plan::step::mps::Encoding;
    type Config = ();

    /// Encodes the fusion's kernel.
    fn run(
        &self,
        step: &crate::plan::MpsStep,
        _: Option<&()>,
        e: &crate::plan::step::mps::Encoding,
    ) -> Result<(), String> {
        let (inputs, output, scratch, keep) =
            (e.inputs.as_slice(), e.output, e.scratch, e.keep.clone());
        let _ = scratch;

        encode(step, inputs, output, scratch, keep)
    }
}
