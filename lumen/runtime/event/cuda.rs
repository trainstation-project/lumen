//! CUDA events backed by `cudaEvent_t` on the default stream.
use std::any::Any;
use std::ffi::c_void;
use std::sync::Arc;

use super::{Event, Recorded, timed};
use crate::device::Device;
use crate::device::cuda::CUDA;
use crate::stream::cuda::ffi;

/// `cudaErrorNotReady`, returned by `cudaEventQuery` while work is pending.
const NOT_READY: i32 = 600;
/// `cudaEventDisableTiming` flag.
const DISABLE_TIMING: u32 = 2;

/// Event on one CUDA device.
pub(super) struct CudaEvent {
    index: usize,
    timing: bool,
    /// The CUDA event, created on first record.
    event: Option<*mut c_void>,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Unrecorded event on device `index`; `timing` allows `elapsed_time` on it.
    pub(super) fn new(index: usize, timing: bool) -> Self {
        CudaEvent {
            index,
            timing,
            event: None,
        }
    }
}

/// Turns a CUDA return code into an error naming `call`.
fn checked(err: i32, call: &str) -> Result<(), String> {
    match err {
        0 => Ok(()),
        e => Err(format!("{call} failed (error {e})")),
    }
}

impl Event for CudaEvent {
    fn device(&self) -> Device {
        Device::Cuda(self.index)
    }

    fn timing(&self) -> bool {
        self.timing
    }

    fn record(&mut self) -> Result<(), String> {
        CUDA.set_device(self.index);

        let event = match self.event {
            Some(event) => event,
            None => {
                let mut event = std::ptr::null_mut();
                let flags = if self.timing { 0 } else { DISABLE_TIMING };

                checked(
                    unsafe { ffi::cudaEventCreateWithFlags(&mut event, flags) },
                    "cudaEventCreateWithFlags",
                )?;

                self.event = Some(event);
                event
            }
        };

        checked(
            unsafe { ffi::cudaEventRecord(event, std::ptr::null_mut()) },
            "cudaEventRecord",
        )
    }

    fn marks(&self) -> Option<&Arc<Recorded>> {
        None
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn recorded(&self) -> bool {
        self.event.is_some()
    }

    fn query(&self) -> bool {
        self.event
            .is_none_or(|event| unsafe { ffi::cudaEventQuery(event) } != NOT_READY)
    }

    fn synchronize(&self) -> Result<(), String> {
        match self.event {
            Some(event) => checked(
                unsafe { ffi::cudaEventSynchronize(event) },
                "cudaEventSynchronize",
            ),
            None => Ok(()),
        }
    }

    fn elapsed_time(&self, end: &dyn Event) -> Result<f64, String> {
        timed(self, end)?;

        let end = end
            .as_any()
            .downcast_ref::<CudaEvent>()
            .expect("a CUDA event, as on one device");

        let (Some(start), Some(stop)) = (self.event, end.event) else {
            unreachable!("both recorded")
        };

        let mut ms = 0f32;

        checked(
            unsafe { ffi::cudaEventElapsedTime(&mut ms, start, stop) },
            "cudaEventElapsedTime",
        )?;

        Ok(ms as f64)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let Some(event) = self.event {
            unsafe { ffi::cudaEventDestroy(event) };
        }
    }
}
