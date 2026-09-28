//! A profiler modeled on PyTorch's (`torch.profiler`, `torch/csrc/profiler`).
//!
//! While a session runs ([`start`] .. [`stop`]) it collects:
//! - **ops**: every tensor op (`lumen::zeros`, `lumen::fill_`, ...) as a
//!   timed range on the CPU clock, nested per thread (PyTorch:
//!   `RecordFunction`), plus user ranges from [`record_function`];
//! - **memory** (`profile_memory`): each allocation and free by the CPU and
//!   caching allocators, with the allocator's totals (PyTorch: `[memory]`
//!   events);
//! - **GPU activity** (`Activity::Cuda` / `Activity::Mps`): device-side
//!   start/end of copies and fills, timed with CUDA events or Metal command
//!   buffer timestamps and correlated with the op that issued them
//!   (PyTorch: Kineto's `gpu_memcpy` / `gpu_memset` activities).
//!
//! The result is a [`Profile`]: raw events, PyTorch-style `key_averages`
//! tables, and a Chrome trace.
//!
//! One session runs at a time, process-wide. When none runs, every hook is
//! a single relaxed atomic load.

mod chrome;
#[cfg(feature = "python")]
pub(crate) mod python;
mod summary;
#[cfg(test)]
mod tests;

pub use summary::{EventAvg, Profile, SortBy};

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::device::Device;

/// What a session records (PyTorch: `ProfilerActivity`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activity {
    /// Ops and user ranges on the CPU clock.
    Cpu,
    /// Device-side timing of CUDA copies and fills.
    Cuda,
    /// Device-side timing of Metal (MPS) fills.
    Mps,
}

/// Session options (PyTorch: `torch.profiler.profile(...)` arguments).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilerConfig {
    pub activities: Vec<Activity>,
    /// Record allocations and frees (PyTorch: `profile_memory`).
    pub profile_memory: bool,
    /// Record the shapes each op was called with (PyTorch: `record_shapes`).
    pub record_shapes: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        ProfilerConfig {
            activities: vec![Activity::Cpu],
            profile_memory: false,
            record_shapes: false,
        }
    }
}

/// What an [`Event`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventKind {
    /// A lumen tensor op (Chrome trace category `cpu_op`).
    Op,
    /// A [`record_function`] range (`user_annotation`).
    UserRange,
    /// An allocation (`bytes > 0`) or free (`bytes < 0`) (`[memory]`).
    Memory,
    /// Device-side work: a copy or fill (`gpu_memcpy` / `gpu_memset`).
    Gpu,
}

/// One recorded event. Times are nanoseconds since the session started.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub id: u64,
    pub name: String,
    pub kind: EventKind,
    pub start_ns: u64,
    /// Equal to `start_ns` for memory events.
    pub end_ns: u64,
    /// The recording thread (a small per-process id).
    pub thread: u64,
    /// The op or range this event ran inside, on the same thread; for GPU
    /// events, the op that issued the work (PyTorch: correlation id).
    pub parent: Option<u64>,
    /// `Cpu` for ops; the memory's or GPU's device otherwise.
    pub device: Device,
    /// Input shapes, with `record_shapes`.
    pub shapes: Vec<Vec<usize>>,
    /// Memory events: bytes allocated (positive) or freed (negative).
    pub bytes: i64,
    /// Memory events: the block's address.
    pub addr: usize,
    /// Memory events: the allocator's allocated and reserved bytes after it.
    pub total_allocated: usize,
    pub total_reserved: usize,
}

impl Event {
    pub fn duration_ns(&self) -> u64 {
        self.end_ns - self.start_ns
    }
}

// ---------------- session state ----------------

// What the current session records, as bits: checked on every hook.
const CPU: u8 = 1;
const MEMORY: u8 = 2;
const CUDA: u8 = 4;
const MPS: u8 = 8;
const SHAPES: u8 = 16;

static FLAGS: AtomicU8 = AtomicU8::new(0);
/// Increments per session, so events from a previous one are dropped.
static SESSION: AtomicU64 = AtomicU64::new(0);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_THREAD: AtomicU64 = AtomicU64::new(1);

struct Session {
    id: u64,
    config: ProfilerConfig,
    start_ns: u64,
    events: Vec<Event>,
}

static STATE: Mutex<Option<Session>> = Mutex::new(None);

thread_local! {
    /// Ids of the ops and ranges open on this thread, innermost last.
    static STACK: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    static THREAD_ID: u64 = NEXT_THREAD.fetch_add(1, Ordering::Relaxed);
}

/// The profiler clock: nanoseconds since the first time it was read.
pub(crate) fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn flags() -> u8 {
    FLAGS.load(Ordering::Relaxed)
}

/// Whether a session is running.
pub fn is_enabled() -> bool {
    flags() != 0
}

/// Whether the session records memory events.
pub(crate) fn memory_enabled() -> bool {
    flags() & MEMORY != 0
}

/// Whether the session times device work on `device`.
// Called by the Metal and CUDA backends, which are compiled only when linked.
#[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn device_enabled(device: Device) -> bool {
    let bit = match device {
        Device::Cpu => return false,
        Device::Cuda(_) => CUDA,
        Device::Mps => MPS,
    };
    flags() & bit != 0
}

/// The running session's id (0 when none): device timers use it to know
/// when their clock alignment is stale.
#[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn session_id() -> u64 {
    if is_enabled() {
        SESSION.load(Ordering::Relaxed)
    } else {
        0
    }
}

/// Start a session. Errors if one is already running (PyTorch allows one
/// profiler at a time too).
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
        };
    }
    if config.profile_memory {
        bits |= MEMORY;
    }
    if config.record_shapes {
        bits |= SHAPES;
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

/// Stop the running session and return what it recorded.
pub fn stop() -> Result<Profile, String> {
    // Let in-flight GPU work finish, so its events are recorded (PyTorch's
    // profiler synchronizes its MPS streams too).
    #[cfg(lumen_mps_linked)]
    crate::stream::mps::synchronize();
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    FLAGS.store(0, Ordering::Relaxed);
    let session = state.take().ok_or("no profiler session is running")?;
    let mut events = session.events;
    // Events are pushed as they end; order them by start for readers.
    events.sort_by_key(|e| (e.start_ns, e.id));
    Ok(Profile::new(events, session.config))
}

/// Add an event to the running session, if it is still session `session`.
fn push(session: u64, mut event: Event) {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = state.as_mut().filter(|s| s.id == session) {
        // Times are relative to the session start; clamp work that began
        // just before it.
        event.start_ns = event.start_ns.saturating_sub(s.start_ns);
        event.end_ns = event.end_ns.saturating_sub(s.start_ns);
        s.events.push(event);
    }
}

fn thread_id() -> u64 {
    THREAD_ID.with(|t| *t)
}

/// The innermost op or range open on this thread.
fn current_parent() -> Option<u64> {
    STACK.with(|s| s.borrow().last().copied())
}

// ---------------- ranges ----------------

/// An open op or range; recorded when dropped (PyTorch: `RecordFunction`).
#[must_use = "the range ends when the guard is dropped"]
pub struct RecordGuard {
    open: Option<OpenRange>,
}

struct OpenRange {
    id: u64,
    session: u64,
    name: String,
    kind: EventKind,
    start_ns: u64,
    parent: Option<u64>,
    shapes: Vec<Vec<usize>>,
}

fn open_range(
    name: impl FnOnce() -> String,
    kind: EventKind,
    shapes: impl FnOnce() -> Vec<Vec<usize>>,
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
            shapes: if bits & SHAPES != 0 {
                shapes()
            } else {
                Vec::new()
            },
        }),
    }
}

impl Drop for RecordGuard {
    fn drop(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let end_ns = now_ns();
        STACK.with(|s| {
            let mut stack = s.borrow_mut();
            // Usually the top; tolerate guards dropped out of order.
            if let Some(i) = stack.iter().rposition(|&id| id == open.id) {
                stack.remove(i);
            }
        });
        let thread = thread_id();
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
                device: Device::Cpu,
                shapes: open.shapes,
                bytes: 0,
                addr: 0,
                total_allocated: 0,
                total_reserved: 0,
            },
        );
    }
}

/// Time a user-defined range until the guard drops (PyTorch:
/// `torch.profiler.record_function`). Ops inside it nest under it.
pub fn record_function(name: impl Into<String>) -> RecordGuard {
    open_range(|| name.into(), EventKind::UserRange, Vec::new)
}

/// Time a lumen op until the guard drops; `shapes` is only called with
/// `record_shapes`.
pub(crate) fn record_op(
    name: &'static str,
    shapes: impl FnOnce() -> Vec<Vec<usize>>,
) -> RecordGuard {
    open_range(|| name.to_owned(), EventKind::Op, shapes)
}

// ---------------- memory and device hooks ----------------

/// Report an allocation (`bytes > 0`) or free (`bytes < 0`) on `device`,
/// with the allocator's totals afterwards.
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
            shapes: Vec::new(),
            bytes,
            addr,
            total_allocated,
            total_reserved,
        },
    );
}

/// Record device work on `device` that ran from `start_ns` to `end_ns` on
/// the profiler clock ([`now_ns`]), issued by the current op.
// Called by the Metal and CUDA backends, which are compiled only when linked.
#[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn record_gpu(name: &'static str, device: Device, start_ns: u64, end_ns: u64) {
    if let Some(context) = gpu_context(device) {
        record_gpu_in(context, name, device, start_ns, end_ns);
    }
}

/// Who submitted device work, captured at submission, for work whose times
/// arrive later on another thread (asynchronous MPS work).
#[derive(Debug, Clone, Copy)]
pub(crate) struct GpuContext {
    session: u64,
    parent: Option<u64>,
    thread: u64,
}

/// The current op's context for device work on `device`, if the session
/// times it.
#[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn gpu_context(device: Device) -> Option<GpuContext> {
    device_enabled(device).then(|| GpuContext {
        session: SESSION.load(Ordering::Relaxed),
        parent: current_parent(),
        thread: thread_id(),
    })
}

/// Record device work captured in `context`.
#[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn record_gpu_in(
    context: GpuContext,
    name: &'static str,
    device: Device,
    start_ns: u64,
    end_ns: u64,
) {
    push(
        context.session,
        Event {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            name: name.to_owned(),
            kind: EventKind::Gpu,
            start_ns,
            end_ns: end_ns.max(start_ns),
            thread: context.thread,
            parent: context.parent,
            device,
            shapes: Vec::new(),
            bytes: 0,
            addr: 0,
            total_allocated: 0,
            total_reserved: 0,
        },
    );
}
