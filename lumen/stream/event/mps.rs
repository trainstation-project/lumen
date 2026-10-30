//! MPS events: recording queues a mark on the Metal stream that completes with its command buffer.
use std::any::Any;
use std::sync::Arc;

use super::{Event, Recorded};
use crate::device::Device;

/// Event on the MPS device.
pub(super) struct MpsEvent {
    timing: bool,
    /// Completion of the last record, set when its command buffer finishes.
    marks: Option<Arc<Recorded>>,
}

impl MpsEvent {
    /// Unrecorded event; `timing` allows `elapsed_time` on it.
    pub(super) fn new(timing: bool) -> Self {
        MpsEvent {
            timing,
            marks: None,
        }
    }
}

impl Event for MpsEvent {
    fn device(&self) -> Device {
        Device::Mps(0)
    }

    fn timing(&self) -> bool {
        self.timing
    }

    fn record(&mut self) -> Result<(), String> {
        let marks = Arc::new(Recorded::default());
        crate::kernels::mps::mark_event(marks.clone())?;
        self.marks = Some(marks);
        Ok(())
    }

    fn marks(&self) -> Option<&Arc<Recorded>> {
        self.marks.as_ref()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Flushes the stream so the mark's command buffer runs, then waits for it.
    fn synchronize(&self) -> Result<(), String> {
        let Some(marks) = &self.marks else {
            return Ok(());
        };

        crate::stream::mps::flush();
        marks.wait().map(|_| ())
    }
}
