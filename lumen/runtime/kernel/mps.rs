//! Launching Metal kernels: each dispatched on the MPS stream through its cached pipeline, or recorded inside
//! `capture` so a plan can replay it against new buffer addresses; and the event markers streams record.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Mutex, PoisonError};

use super::Launch;
use crate::Tensor;
use crate::codegen::utils::mps::Mps;
use crate::runtime::mps::Grid;
use crate::stream::mps::{self, Completion};

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
impl Launch for Mps {
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
