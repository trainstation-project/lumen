use std::any::Any;
use std::sync::Arc;

use super::{Event, Recorded};
use crate::device::Device;

pub(super) struct CpuEvent {
    timing: bool,
    marks: Option<Arc<Recorded>>,
}

impl CpuEvent {
    pub(super) fn new(timing: bool) -> Self {
        CpuEvent {
            timing,
            marks: None,
        }
    }
}

impl Event for CpuEvent {
    fn device(&self) -> Device {
        Device::Cpu(0)
    }

    fn timing(&self) -> bool {
        self.timing
    }

    fn record(&mut self) -> Result<(), String> {
        self.marks = Some(Recorded::now());
        Ok(())
    }

    fn marks(&self) -> Option<&Arc<Recorded>> {
        self.marks.as_ref()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
