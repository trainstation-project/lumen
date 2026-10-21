use std::cell::{Cell, RefCell};

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

pub(crate) fn capture() {
    CAPTURED.with_borrow_mut(|c| *c = Some(Vec::new()));
}

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
