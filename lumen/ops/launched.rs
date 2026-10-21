use std::cell::{Cell, RefCell};

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) struct Launched {
    pub(crate) step: usize,
    pub(crate) label: &'static str,
    pub(crate) kernel: String,
    pub(crate) code: Option<String>,
}

thread_local! {
    static CAPTURED: RefCell<Option<Vec<Launched>>> = const { RefCell::new(None) };
    static STEP: Cell<(usize, &'static str)> = const { Cell::new((0, "")) };
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn capture() {
    CAPTURED.with_borrow_mut(|c| *c = Some(Vec::new()));
}

#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn captured() -> Vec<Launched> {
    CAPTURED.with_borrow_mut(Option::take).unwrap_or_default()
}

pub(crate) fn step(index: usize, label: &'static str) {
    STEP.set((index, label));
}

pub(crate) fn launched(kernel: &str, code: impl FnOnce() -> Option<String>) {
    CAPTURED.with_borrow_mut(|c| {
        if let Some(c) = c {
            let (step, label) = STEP.get();
            c.push(Launched {
                step,
                label,
                kernel: kernel.to_owned(),
                code: code(),
            });
        }
    });
}
