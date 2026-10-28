//! The profiler: a global session that records op and user ranges, memory events and device kernel
//! timings as nested events, summarized by `Profile` and exportable as a Chrome trace.
mod chrome;
#[cfg(lumen_cupti_linked)]
pub(crate) mod cupti;
#[cfg(feature = "python")]
pub(crate) mod python;
mod summary;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;

pub use summary::{EventAvg, Profile, SortBy};

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::DType;
use crate::device::Device;
use crate::graph::TensorType;

/// What a profiler session records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activity {
    /// Host-side op and user ranges.
    Cpu,
    /// CUDA kernels and copies, timed with CUPTI.
    Cuda,
    /// Metal kernels.
    Mps,
    /// TPU kernels.
    Tpu,
}

/// Options for a profiler session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilerConfig {
    pub activities: Vec<Activity>,
    /// Record allocations and frees as memory events.
    pub profile_memory: bool,
    /// Record each op's input and output types and each kernel's strides.
    pub record_shapes: bool,
}

impl Default for ProfilerConfig {
    /// Records CPU activity only, without memory or shapes.
    fn default() -> Self {
        ProfilerConfig {
            activities: vec![Activity::Cpu],
            profile_memory: false,
            record_shapes: false,
        }
    }
}

/// What a recorded event stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A lumen op on the host.
    Op,
    /// A `record_function` range.
    UserRange,
    /// An allocation (positive `bytes`) or free (negative).
    Memory,
    /// A kernel or copy timed on a device.
    Gpu,
    /// A plan step run on the host, inside its op.
    HostKernel,
}

/// One recorded event; times are nanoseconds since the session started.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Unique id, referenced by children's `parent`.
    pub id: u64,
    pub name: String,
    pub kind: EventKind,
    pub start_ns: u64,
    pub end_ns: u64,
    /// Profiler-assigned id of the recording thread.
    pub thread: u64,
    /// Id of the enclosing range or launching op.
    pub parent: Option<u64>,
    pub device: Device,
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<TensorType>,
    /// Strides in elements the kernel reads each input with.
    pub input_strides: Vec<Vec<usize>>,
    /// Strides in elements the kernel writes each output with.
    pub output_strides: Vec<Vec<usize>>,
    /// Dtypes the op accumulates in.
    pub accum: Vec<DType>,
    /// Bytes allocated (positive) or freed (negative), for memory events.
    pub bytes: i64,
    pub addr: usize,
    /// Device bytes allocated after a memory event.
    pub total_allocated: usize,
    /// Device bytes reserved by the allocator after a memory event.
    pub total_reserved: usize,
    /// Name of the device kernel that ran, when it differs from the event name.
    pub kernel: Option<String>,
}

impl Event {
    /// End minus start time.
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }
}

/// Bits of `FLAGS`, one per thing the running session records.
const CPU: u8 = 1;
const MEMORY: u8 = 2;
const CUDA: u8 = 4;
const MPS: u8 = 8;
const SHAPES: u8 = 16;
const TPU: u8 = 32;

/// The running session's flags; zero when no session is running.
static FLAGS: AtomicU8 = AtomicU8::new(0);
/// Id of the latest session, so late events from an earlier session are dropped.
static SESSION: AtomicU64 = AtomicU64::new(0);
/// Next event id.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Next profiler thread id.
static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);

/// A running profiler session and the events recorded so far.
struct Session {
    id: u64,
    config: ProfilerConfig,
    start_ns: u64,
    events: Vec<Event>,
}

/// The running session, if any.
static STATE: Mutex<Option<Session>> = Mutex::new(None);

thread_local! {
    /// Ids of this thread's open ranges, innermost last.
    static STACK: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    /// This thread's profiler id.
    static THREAD_ID: u64 = NEXT_THREAD.fetch_add(1, Ordering::Relaxed);
    /// Name and types for the next device launch on this thread, set by the step about to launch it.
    static NEXT_LAUNCH: RefCell<Option<Launch>> = const { RefCell::new(None) };
    /// Set inside `untraced`, hiding this thread's work from the profiler.
    static UNTRACED: Cell<bool> = const { Cell::new(false) };
}

/// Runs `f` with profiling off on this thread, restoring the previous state afterwards.
pub(crate) fn untraced<R>(f: impl FnOnce() -> R) -> R {
    /// Restores the saved `UNTRACED` value on drop, even on panic.
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            UNTRACED.set(self.0);
        }
    }
    let _restore = Restore(UNTRACED.replace(true));
    f()
}

/// False inside `untraced`.
pub(crate) fn traced() -> bool {
    !UNTRACED.get()
}

/// Types and strides of a step's inputs and outputs, attached to its kernel events.
#[derive(Debug, Clone, Default)]
pub(crate) struct IoTypes {
    pub inputs: Vec<TensorType>,
    pub outputs: Vec<TensorType>,
    pub input_strides: Vec<Vec<usize>>,
    pub output_strides: Vec<Vec<usize>>,
}

/// Name and optional types for a pending device launch.
#[derive(Debug, Clone)]
pub(crate) struct Launch {
    name: &'static str,
    types: Option<IoTypes>,
}

/// Labels this thread's next device launch with `name` (and its types when recording shapes).
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) fn next_launch(name: &'static str, types: impl FnOnce() -> IoTypes) {
    let bits = flags();
    if bits & (MPS | CUDA) != 0 {
        let types = (bits & SHAPES != 0).then(types);
        NEXT_LAUNCH.with(|l| *l.borrow_mut() = Some(Launch { name, types }));
    }
}

/// Drops a pending launch label, for launches that are skipped.
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) fn clear_next_launch() {
    NEXT_LAUNCH.with(|l| l.borrow_mut().take());
}

/// Monotonic nanoseconds since the first call in this process.
pub(crate) fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// The session flags, or zero on a thread inside `untraced`.
fn flags() -> u8 {
    match traced() {
        true => FLAGS.load(Ordering::Relaxed),
        false => 0,
    }
}

/// True when a profiler session is recording on this thread.
pub fn is_enabled() -> bool {
    flags() != 0
}

/// True when the session records memory events.
pub(crate) fn memory_enabled() -> bool {
    flags() & MEMORY != 0
}

/// True when the session times kernels on `device`.
#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cupti_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
pub(crate) fn device_enabled(device: Device) -> bool {
    let bit = match device {
        Device::Cpu(_) | Device::Meta(_) => return false,
        Device::Cuda(_) => CUDA,
        Device::Mps(_) => MPS,
        Device::Tpu(_) => TPU,
    };
    flags() & bit != 0
}

/// Starts a profiler session; errors if one is already running or an activity is not available.
pub fn start(config: ProfilerConfig) -> Result<(), String> {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if state.is_some() {
        return Err("a profiler session is already running".to_owned());
    }
    let mut bits = 0;
    for activity in &config.activities {
        bits |= match activity {
            Activity::Cpu => CPU,
            Activity::Cuda => CUDA,
            Activity::Mps => MPS,
            Activity::Tpu => TPU,
        };
    }
    if config.profile_memory {
        bits |= MEMORY;
    }
    if config.record_shapes {
        bits |= SHAPES;
    }
    if bits & TPU != 0 {
        #[cfg(lumen_tpu_linked)]
        tpu::start()?;
        #[cfg(not(lumen_tpu_linked))]
        return Err("timing TPUs needs lumen built with the TPU backend".to_owned());
    }
    #[cfg(lumen_cupti_linked)]
    if bits & CUDA != 0 {
        cupti::start();
    }
    let id = SESSION.fetch_add(1, Ordering::Relaxed) + 1;
    *state = Some(Session {
        id,
        config,
        start_ns: now_ns(),
        events: Vec::new(),
    });
    FLAGS.store(bits, Ordering::Relaxed);
    Ok(())
}

/// Ends the session, collects device timings, and returns the profile with each kernel event
/// given its launching op's types and accumulation dtypes.
pub fn stop() -> Result<Profile, String> {
    #[cfg(lumen_mps_linked)]
    crate::stream::mps::synchronize();
    #[cfg(lumen_cupti_linked)]
    cupti::stop();
    #[cfg(lumen_tpu_linked)]
    if let Err(e) = tpu::stop(op_name) {
        eprintln!("{}: the TPU profile: {e}", crate::LIBRARY_NAME);
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    FLAGS.store(0, Ordering::Relaxed);
    let session = state.take().ok_or("no profiler session is running")?;
    let mut events = session.events;
    events.sort_by_key(|e| (e.start_ns, e.id));
    // Kernels without their own types take their parent op's.
    let types: std::collections::HashMap<u64, (IoTypes, Vec<DType>)> = events
        .iter()
        .filter(|e| e.kind == EventKind::Op)
        .map(|e| {
            let io = IoTypes {
                inputs: e.inputs.clone(),
                outputs: e.outputs.clone(),
                input_strides: e.input_strides.clone(),
                output_strides: e.output_strides.clone(),
            };
            (e.id, (io, e.accum.clone()))
        })
        .collect();
    for e in events
        .iter_mut()
        .filter(|e| matches!(e.kind, EventKind::Gpu | EventKind::HostKernel))
    {
        if let Some((io, accum)) = e.parent.and_then(|p| types.get(&p)) {
            if e.inputs.is_empty() && e.outputs.is_empty() {
                (e.inputs, e.outputs) = (io.inputs.clone(), io.outputs.clone());
                (e.input_strides, e.output_strides) =
                    (io.input_strides.clone(), io.output_strides.clone());
            }
            e.accum = accum.clone();
        }
    }
    Ok(Profile::new(events, session.config))
}

/// Name of recorded event `id`, for naming TPU kernels after their op.
#[cfg(lumen_tpu_linked)]
fn op_name(id: u64) -> Option<String> {
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let events = &state.as_ref()?.events;
    events
        .iter()
        .rev()
        .find(|e| e.id == id)
        .map(|e| e.name.clone())
}

/// Adds `event` to session `session`, rebasing its times on the session start; dropped if that session ended.
fn push(session: u64, mut event: Event) {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = state.as_mut().filter(|s| s.id == session) {
        event.start_ns = event.start_ns.saturating_sub(s.start_ns);
        event.end_ns = event.end_ns.saturating_sub(s.start_ns);
        s.events.push(event);
    }
}

/// This thread's profiler id.
fn thread_id() -> u64 {
    THREAD_ID.with(|t| *t)
}

/// Id of this thread's innermost open range.
fn current_parent() -> Option<u64> {
    STACK.with(|s| s.borrow().last().copied())
}

/// Guard for an open range; the event is recorded when it drops.
#[must_use = "the range ends when the guard is dropped"]
pub struct RecordGuard {
    open: Option<OpenRange>,
}

/// A range that has started but not yet ended.
struct OpenRange {
    id: u64,
    session: u64,
    name: String,
    kind: EventKind,
    start_ns: u64,
    parent: Option<u64>,
    /// Input, output and accumulation types, when recording shapes.
    types: Option<(Vec<TensorType>, Vec<TensorType>, Vec<DType>)>,
    /// Input and output strides.
    strides: (Vec<Vec<usize>>, Vec<Vec<usize>>),
}

/// Opens a range of `kind` nested under the current one; a no-op guard when CPU activity is off.
fn open_range(
    name: impl FnOnce() -> String,
    kind: EventKind,
    inputs: impl FnOnce() -> Vec<TensorType>,
) -> RecordGuard {
    let bits = flags();
    if bits & CPU == 0 {
        return RecordGuard { open: None };
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let parent = current_parent();
    STACK.with(|s| s.borrow_mut().push(id));
    RecordGuard {
        open: Some(OpenRange {
            id,
            session: SESSION.load(Ordering::Relaxed),
            name: name(),
            kind,
            start_ns: now_ns(),
            parent,
            types: (bits & SHAPES != 0).then(|| (inputs(), Vec::new(), Vec::new())),
            strides: (Vec::new(), Vec::new()),
        }),
    }
}

impl RecordGuard {
    /// Sets the range's output types, when recording shapes.
    pub(crate) fn outputs(&mut self, outputs: impl FnOnce() -> Vec<TensorType>) {
        if let Some((_, out, _)) = self.open.as_mut().and_then(|o| o.types.as_mut()) {
            *out = outputs();
        }
    }

    /// Sets the range's input and output strides, when recording shapes.
    pub(crate) fn strides(&mut self, strides: impl FnOnce() -> (Vec<Vec<usize>>, Vec<Vec<usize>>)) {
        if let Some(open) = self.open.as_mut().filter(|o| o.types.is_some()) {
            open.strides = strides();
        }
    }

    /// Sets the range's accumulation dtypes, when recording shapes.
    pub(crate) fn accum(&mut self, accum: impl FnOnce() -> Vec<DType>) {
        if let Some((_, _, a)) = self.open.as_mut().and_then(|o| o.types.as_mut()) {
            *a = accum();
        }
    }
}

impl Drop for RecordGuard {
    /// Closes the range and records it as an event.
    fn drop(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let end_ns = now_ns();
        STACK.with(|s| {
            let mut stack = s.borrow_mut();
            if let Some(i) = stack.iter().rposition(|&id| id == open.id) {
                stack.remove(i);
            }
        });
        let thread = thread_id();
        let (inputs, outputs, accum) = open.types.unwrap_or_default();
        let (input_strides, output_strides) = open.strides;
        push(
            open.session,
            Event {
                id: open.id,
                name: open.name,
                kind: open.kind,
                start_ns: open.start_ns,
                end_ns,
                thread,
                parent: open.parent,
                device: Device::Cpu(0),
                inputs,
                outputs,
                input_strides,
                output_strides,
                accum,
                bytes: 0,
                addr: 0,
                total_allocated: 0,
                total_reserved: 0,
                kernel: None,
            },
        );
    }
}

/// Opens a user range named `name`.
pub fn record_function(name: impl Into<String>) -> RecordGuard {
    open_range(|| name.into(), EventKind::UserRange, Vec::new)
}

/// Opens a range for a plan step running on the host.
pub(crate) fn record_host_kernel(name: &'static str) -> RecordGuard {
    open_range(|| name.to_owned(), EventKind::HostKernel, Vec::new)
}

/// Opens a range for an op, recording its input types when shapes are on.
pub(crate) fn record_op(
    name: &'static str,
    inputs: impl FnOnce() -> Vec<TensorType>,
) -> RecordGuard {
    open_range(|| name.to_owned(), EventKind::Op, inputs)
}

/// Records a memory event for an allocation or free on `device`, when memory profiling is on.
pub(crate) fn report_memory(
    device: Device,
    addr: usize,
    bytes: i64,
    total_allocated: usize,
    total_reserved: usize,
) {
    if flags() & MEMORY == 0 {
        return;
    }
    let (session, now, parent, thread) = (
        SESSION.load(Ordering::Relaxed),
        now_ns(),
        current_parent(),
        thread_id(),
    );
    push(
        session,
        Event {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            name: "[memory]".to_owned(),
            kind: EventKind::Memory,
            start_ns: now,
            end_ns: now,
            thread,
            parent,
            device,
            inputs: Vec::new(),
            outputs: Vec::new(),
            input_strides: Vec::new(),
            output_strides: Vec::new(),
            accum: Vec::new(),
            bytes,
            addr,
            total_allocated,
            total_reserved,
            kernel: None,
        },
    );
}

/// Records a device kernel under the current context (tests only).
#[cfg(test)]
pub(crate) fn record_gpu(name: &'static str, device: Device, start_ns: u64, end_ns: u64) {
    if let Some(context) = gpu_context(device) {
        record_gpu_in(context, name, device, start_ns, end_ns);
    }
}

/// Where a device launch was issued from, captured at launch time to record its timing later.
#[derive(Debug, Clone)]
pub(crate) struct GpuContext {
    session: u64,
    parent: Option<u64>,
    thread: u64,
    launch: Option<Launch>,
}

/// Captures the context for a launch on `device`, consuming any pending launch label; `None` when the device is not profiled.
#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cupti_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
pub(crate) fn gpu_context(device: Device) -> Option<GpuContext> {
    let launch = NEXT_LAUNCH.with(|l| l.borrow_mut().take());
    device_enabled(device).then(|| GpuContext {
        session: SESSION.load(Ordering::Relaxed),
        parent: current_parent(),
        thread: thread_id(),
        launch,
    })
}

/// Records a device kernel or copy timed from `start_ns` to `end_ns` under `context` (tests only).
#[cfg(test)]
pub(crate) fn record_gpu_in(
    context: GpuContext,
    name: &str,
    device: Device,
    start_ns: u64,
    end_ns: u64,
) {
    record_kernel_in(context, name, None, device, start_ns, end_ns);
}

/// Records a device kernel, with `kernel` naming the kernel that ran when it differs from the step's name.
#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cupti_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
pub(crate) fn record_kernel_in(
    context: GpuContext,
    name: &str,
    kernel: Option<String>,
    device: Device,
    start_ns: u64,
    end_ns: u64,
) {
    record_in(
        context,
        name,
        kernel,
        EventKind::Gpu,
        device,
        start_ns,
        end_ns,
    );
}

/// Records a plan step that ran on the host under `context`.
#[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
pub(crate) fn record_host_kernel_in(context: GpuContext, name: &str, start_ns: u64, end_ns: u64) {
    record_in(
        context,
        name,
        None,
        EventKind::HostKernel,
        Device::Cpu(0),
        start_ns,
        end_ns,
    );
}

/// Records a timed event under `context`, named by its launch label when there is one.
fn record_in(
    context: GpuContext,
    name: &str,
    kernel: Option<String>,
    kind: EventKind,
    device: Device,
    start_ns: u64,
    end_ns: u64,
) {
    let (name, types) = match context.launch {
        Some(Launch { name, types }) => (name, types),
        None => (name, None),
    };
    let io = types.unwrap_or_default();
    push(
        context.session,
        Event {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            name: name.to_owned(),
            kind,
            start_ns,
            end_ns: end_ns.max(start_ns),
            thread: context.thread,
            parent: context.parent,
            device,
            inputs: io.inputs,
            outputs: io.outputs,
            input_strides: io.input_strides,
            output_strides: io.output_strides,
            accum: Vec::new(),
            bytes: 0,
            addr: 0,
            total_allocated: 0,
            total_reserved: 0,
            kernel,
        },
    );
}
