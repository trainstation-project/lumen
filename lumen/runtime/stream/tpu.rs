//! The TPU stream, synchronized through the TPU allocator, and the batches its launches collect in to run as one
//! executable.

use std::cell::RefCell;

use super::Stream;
use crate::runtime::kernel::tpu::{Lowered, run};

/// lumen's stream on TPU device `index`.
pub struct TpuStream {
    pub index: usize,
}

impl Stream for TpuStream {
    /// Blocks until the TPU's queued work finishes, panicking if the wait fails.
    fn synchronize(&self) {
        crate::allocator::tpu::wait(self.index).unwrap_or_else(|e| panic!("{e}"))
    }
}

// Per thread, the stack of open batches: device index and the launches collected so far.
thread_local! {
    pub(crate) static BATCHES: RefCell<Vec<(usize, Vec<Lowered>)>> = const { RefCell::new(Vec::new()) };
}

/// Collects TPU launches on one device so they run as a single executable; dropping it unrun
/// discards them.
pub(crate) struct Batch {
    /// Whether `run` consumed the batch.
    ran: bool,
}

impl Batch {
    /// Opens a batch on device `index`.
    pub(crate) fn new(index: usize) -> Batch {
        BATCHES.with(|b| b.borrow_mut().push((index, Vec::new())));
        Batch { ran: false }
    }

    /// Runs every launch collected in the batch.
    pub(crate) fn run(mut self) -> Result<(), String> {
        self.ran = true;
        let (index, launches) = BATCHES.with(|b| b.borrow_mut().pop()).expect("a batch");
        run(index, &launches, &[])
    }
}

/// Runs the launches pending in the innermost batch now, when it is on device `index`.
pub(crate) fn flush(index: usize) -> Result<(), String> {
    let pending = BATCHES.with(|b| match b.borrow_mut().last_mut() {
        Some((i, launches)) if *i == index => std::mem::take(launches),
        _ => Vec::new(),
    });

    match pending.is_empty() {
        true => Ok(()),
        false => run(index, &pending, &[]),
    }
}

impl Drop for Batch {
    /// Discards the batch's launches unless it ran.
    fn drop(&mut self) {
        if !self.ran {
            BATCHES.with(|b| b.borrow_mut().pop());
        }
    }
}
