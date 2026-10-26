//! Per-thread record of kernels launched while the profiler traces, with their generated source for inspection.
use std::cell::{Cell, RefCell};

#[cfg_attr(not(feature = "python"), allow(dead_code))]
/// One kernel launch observed during capture.
pub(crate) struct Launched {
    /// Index of the plan step that launched it.
    pub(crate) step: usize,
    /// Label of that plan step.
    pub(crate) label: &'static str,
    pub(crate) kernel: String,
    /// Generated source, when the backend can show it.
    pub(crate) code: Option<String>,
    /// Template the kernel was instantiated from with its arguments, when it came from one.
    pub(crate) template: Option<(String, String)>,
}

thread_local! {
    /// Launches recorded since `capture`, or `None` when not capturing.
    static CAPTURED: RefCell<Option<Vec<Launched>>> = const { RefCell::new(None) };
    /// Plan step currently running, attributed to each launch.
    static STEP: Cell<(usize, &'static str)> = const { Cell::new((0, "")) };
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
/// Starts recording launches on this thread, discarding any earlier capture.
pub(crate) fn capture() {
    CAPTURED.with_borrow_mut(|c| *c = Some(Vec::new()));
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
/// Stops recording and returns the launches captured on this thread.
pub(crate) fn captured() -> Vec<Launched> {
    CAPTURED.with_borrow_mut(Option::take).unwrap_or_default()
}

/// Marks plan step `index` named `label` as the one launching subsequent kernels.
pub(crate) fn step(index: usize, label: &'static str) {
    STEP.set((index, label));
}

/// Records a launch of `kernel` whose source `code` is built lazily, only when capturing.
pub(crate) fn launched(kernel: &str, code: impl FnOnce() -> Option<String>) {
    templated(kernel, || (code(), None));
}

/// Records a launch of `kernel` whose `(source, template)` pair is built lazily, only when tracing and capturing.
pub(crate) fn templated(
    kernel: &str,
    code: impl FnOnce() -> (Option<String>, Option<(String, String)>),
) {
    if !crate::profiler::traced() {
        return;
    }
    CAPTURED.with_borrow_mut(|c| {
        if let Some(c) = c {
            let (step, label) = STEP.get();
            let (code, template) = code();
            c.push(Launched {
                step,
                label,
                kernel: kernel.to_owned(),
                code,
                template,
            });
        }
    });
}
