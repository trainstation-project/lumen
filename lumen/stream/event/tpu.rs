use std::any::Any;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use super::{Event, Recorded, host_seconds};
use crate::device::Device;
use crate::device::tpu::{error, ffi};

pub(super) struct TpuEvent {
    index: usize,
    timing: bool,
    marks: Option<Arc<Recorded>>,
    ready: Option<NonNull<c_void>>,
}

unsafe impl Send for TpuEvent {}
unsafe impl Sync for TpuEvent {}

impl TpuEvent {
    pub(super) fn new(index: usize, timing: bool) -> Self {
        TpuEvent {
            index,
            timing,
            marks: None,
            ready: None,
        }
    }

    fn release(&mut self) {
        if let Some(ready) = self.ready.take() {
            unsafe { ffi::lumen_tpu_event_free(ready.as_ptr()) };
        }
    }
}

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

    fn record(&mut self) -> Result<(), String> {
        self.release();
        crate::kernels::tpu::flush(self.index)?;
        let Some(ready) = crate::allocator::tpu::ready_event(self.index)? else {
            self.marks = Some(Recorded::now());
            return Ok(());
        };
        let marks = Arc::new(Recorded::default());
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
