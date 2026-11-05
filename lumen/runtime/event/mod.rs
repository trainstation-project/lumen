//! Device events for synchronizing with and timing queued work, one implementation per backend.
mod cpu;
#[cfg(lumen_cuda_linked)]
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
#[cfg(lumen_tpu_linked)]
mod tpu;

use std::any::Any;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

use crate::device::Device;

/// A point in a device's work queue that can be waited on and timed against another event.
pub trait Event: Send + Sync {
    /// Device the event records on.
    fn device(&self) -> Device;
    /// Whether the event was created with timing enabled.
    fn timing(&self) -> bool;
    /// Marks the point after all work queued so far.
    fn record(&mut self) -> Result<(), String>;
    /// Completion times of the last record, for events that report them on the host.
    fn marks(&self) -> Option<&Arc<Recorded>>;
    /// The event as `Any`, for downcasting to its backend type.
    fn as_any(&self) -> &dyn Any;

    /// Whether the event has been recorded.
    fn recorded(&self) -> bool {
        self.marks().is_some()
    }

    /// Whether the work before the event has finished; true for an unrecorded event.
    fn query(&self) -> bool {
        self.marks().is_none_or(|r| r.done())
    }

    /// Blocks until the work before the event finishes; errors if that work failed.
    fn synchronize(&self) -> Result<(), String> {
        match self.marks() {
            Some(marks) => marks.wait().map(|_| ()),
            None => Ok(()),
        }
    }

    /// Milliseconds between this event and `end`, waiting for both.
    fn elapsed_time(&self, end: &dyn Event) -> Result<f64, String> {
        timed(self, end)?;
        self.synchronize()?;
        end.synchronize()?;

        let (Some(start), Some(stop)) = (self.marks(), end.marks()) else {
            unreachable!("both recorded")
        };

        let ((_, from), (to, _)) = (start.wait()?, stop.wait()?);
        Ok((to - from) * 1e3)
    }
}

/// Creates an unrecorded event on `device`.
pub fn event_for(device: Device, timing: bool) -> Result<Box<dyn Event>, String> {
    device.check()?;

    Ok(match device {
        Device::Cpu(_) => Box::new(cpu::CpuEvent::new(timing)),
        Device::Meta(_) => {
            return Err("the meta device runs nothing to record an event after".into());
        }
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => Box::new(mps::MpsEvent::new(timing)),
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => Box::new(cuda::CudaEvent::new(index, timing)),
        #[cfg(lumen_tpu_linked)]
        Device::Tpu(index) => Box::new(tpu::TpuEvent::new(index, timing)),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{device} is never available without its backend"),
    })
}

/// Checks that `start` and `end` can be timed against each other: same device, timing enabled,
/// both recorded.
fn timed(start: &(impl Event + ?Sized), end: &dyn Event) -> Result<(), String> {
    if start.device() != end.device() {
        return Err(format!(
            "elapsed_time: events on {} and {} share no clock",
            start.device(),
            end.device()
        ));
    }

    if !(start.timing() && end.timing()) {
        return Err("elapsed_time: both events need enable_timing=True".into());
    }

    if !(start.recorded() && end.recorded()) {
        return Err("elapsed_time: record both events first".into());
    }

    Ok(())
}

/// Completion of a recorded event, set once by the backend and waited on by the host.
#[derive(Default)]
pub struct Recorded {
    /// Start and end times in seconds and whether the work succeeded; `None` until set.
    times: Mutex<Option<(f64, f64, bool)>>,
    ready: Condvar,
}

impl Recorded {
    /// Records completion and wakes waiters.
    pub(crate) fn set(&self, start: f64, end: f64, ok: bool) {
        *self.times.lock().unwrap_or_else(|e| e.into_inner()) = Some((start, end, ok));
        self.ready.notify_all();
    }

    /// Already-completed marks at the current host time.
    pub(crate) fn now() -> Arc<Recorded> {
        let recorded = Arc::new(Recorded::default());
        let now = host_seconds();
        recorded.set(now, now, true);
        recorded
    }

    /// Whether completion has been recorded.
    fn done(&self) -> bool {
        self.times
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Blocks until completion and returns `(start, end)` seconds, or an error if the work failed.
    fn wait(&self) -> Result<(f64, f64), String> {
        let mut times = self.times.lock().unwrap_or_else(|e| e.into_inner());

        while times.is_none() {
            times = self.ready.wait(times).unwrap_or_else(|e| e.into_inner());
        }

        match times.expect("set") {
            (start, end, true) => Ok((start, end)),
            _ => Err("the work before the event failed on the device".into()),
        }
    }
}

/// Seconds since the first call in this process, as a monotonic host clock.
pub(crate) fn host_seconds() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}
