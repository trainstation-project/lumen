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

pub trait Event: Send + Sync {
    fn device(&self) -> Device;
    fn timing(&self) -> bool;
    fn record(&mut self) -> Result<(), String>;
    fn marks(&self) -> Option<&Arc<Recorded>>;
    fn as_any(&self) -> &dyn Any;

    fn recorded(&self) -> bool {
        self.marks().is_some()
    }

    fn query(&self) -> bool {
        self.marks().is_none_or(|r| r.done())
    }

    fn synchronize(&self) -> Result<(), String> {
        match self.marks() {
            Some(marks) => marks.wait().map(|_| ()),
            None => Ok(()),
        }
    }

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

#[derive(Default)]
pub struct Recorded {
    times: Mutex<Option<(f64, f64, bool)>>,
    ready: Condvar,
}

impl Recorded {
    pub(crate) fn set(&self, start: f64, end: f64, ok: bool) {
        *self.times.lock().unwrap_or_else(|e| e.into_inner()) = Some((start, end, ok));
        self.ready.notify_all();
    }

    pub(crate) fn now() -> Arc<Recorded> {
        let recorded = Arc::new(Recorded::default());
        let now = host_seconds();
        recorded.set(now, now, true);
        recorded
    }

    fn done(&self) -> bool {
        self.times
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

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

pub(crate) fn host_seconds() -> f64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_secs_f64()
}
