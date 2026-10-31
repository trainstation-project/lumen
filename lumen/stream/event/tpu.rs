//! TPU events: recording waits on the PJRT ready event of the device's latest work.
use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use super::{Event, Recorded, host_seconds};
use crate::device::Device;
use crate::device::tpu::{error, ffi};

/// Event on one TPU device.
pub(super) struct TpuEvent {
    index: usize,
    timing: bool,
    /// Completion of the last record, set from the PJRT callback.
    marks: Option<Arc<Recorded>>,
    /// PJRT ready event being waited on, freed on the next record or drop.
    ready: Option<NonNull<c_void>>,
}

unsafe impl Send for TpuEvent {}
unsafe impl Sync for TpuEvent {}

impl TpuEvent {
    /// Unrecorded event on device `index`; `timing` allows `elapsed_time` on it.
    pub(super) fn new(index: usize, timing: bool) -> Self {
        TpuEvent {
            index,
            timing,
            marks: None,
            ready: None,
        }
    }

    /// Frees the held PJRT ready event, if any.
    fn release(&mut self) {
        if let Some(ready) = self.ready.take() {
            unsafe { ffi::lumen_tpu_event_free(ready.as_ptr()) };
        }
    }
}

/// PJRT ready callback: takes back the `Arc<Recorded>` passed as `user` and sets it to now.
unsafe extern "C" fn finished(user: *mut c_void, ok: i32) {
    let marks = unsafe { Arc::from_raw(user.cast::<Recorded>()) };
    let now = host_seconds();
    marks.set(now, now, ok != 0);
}

impl Event for TpuEvent {
    fn device(&self) -> Device {
        Device::Tpu(self.index)
    }

    fn timing(&self) -> bool {
        self.timing
    }

    /// Flushes queued kernels and hooks the device's latest ready event; with no work pending the
    /// event completes immediately.
    fn record(&mut self) -> Result<(), String> {
        self.release();
        crate::runtime::tpu::flush(self.index)?;

        let Some(ready) = crate::allocator::tpu::ready_event(self.index)? else {
            self.marks = Some(Recorded::now());
            return Ok(());
        };

        let marks = Arc::new(Recorded::default());
        // The callback owns one reference to the marks; reclaim it if registration fails.
        let user = Arc::into_raw(marks.clone()).cast_mut().cast::<c_void>();

        if unsafe { ffi::lumen_tpu_on_ready(ready.as_ptr(), finished, user) } != 0 {
            drop(unsafe { Arc::from_raw(user.cast::<Recorded>()) });

            unsafe { ffi::lumen_tpu_event_free(ready.as_ptr()) };

            return Err(error());
        }

        self.ready = Some(ready);
        self.marks = Some(marks);
        Ok(())
    }

    fn marks(&self) -> Option<&Arc<Recorded>> {
        self.marks.as_ref()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for TpuEvent {
    fn drop(&mut self) {
        self.release();
    }
}
