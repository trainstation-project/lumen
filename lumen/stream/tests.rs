//! Tests for creating and synchronizing device streams.

use super::stream_for;
use crate::Device;

/// Every available device yields a stream that synchronizes.
#[test]
fn every_present_device_has_a_stream() {
    let devices = [
        Device::Cpu(0),
        Device::Meta(0),
        Device::Mps(0),
        Device::Cuda(0),
    ];

    for device in devices.into_iter().filter(|d| d.check().is_ok()) {
        stream_for(device).unwrap().synchronize();
    }
}

/// Device indices past the last present device yield an error instead of a stream.
#[test]
fn absent_devices_have_none() {
    let past_last = Device::Cuda(crate::device::cuda::CUDA.device_count());
    let err = stream_for(past_last).err().unwrap();
    assert!(err.contains("is not available"), "{err}");
    assert!(stream_for(Device::Cpu(1)).is_err());
}

/// Events of each present device are unrecorded until recorded, and downcast to their backend's type.
#[test]
fn unrecorded_events_are_done_and_backends_have_their_own_types() {
    use super::event_for;
    let cpu = event_for(Device::Cpu(0), true).unwrap();
    assert!(!cpu.recorded() && cpu.query());
    cpu.synchronize().unwrap();
    let other = event_for(Device::Cpu(0), false).unwrap();
    assert_eq!(cpu.as_any().type_id(), other.as_any().type_id());
    if Device::Mps(0).check().is_ok() {
        let mut mps = event_for(Device::Mps(0), true).unwrap();
        assert!(!mps.recorded() && mps.query());
        mps.synchronize().unwrap();
        assert_ne!(mps.as_any().type_id(), cpu.as_any().type_id());
        // Events on different devices have no common clock.
        mps.record().unwrap();
        let mut cpu = event_for(Device::Cpu(0), true).unwrap();
        cpu.record().unwrap();
        assert_eq!(
            mps.elapsed_time(cpu.as_ref()).unwrap_err(),
            "elapsed_time: events on mps:0 and cpu:0 share no clock"
        );
    }
    assert_eq!(
        event_for(Device::Meta(0), false).err().unwrap(),
        "the meta device runs nothing to record an event after"
    );
}

/// An event whose marks were set as failed, standing in for a backend whose work errored.
struct Failed(std::sync::Arc<super::event::Recorded>);

impl super::Event for Failed {
    fn device(&self) -> Device {
        Device::Cpu(0)
    }
    fn timing(&self) -> bool {
        true
    }
    fn record(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn marks(&self) -> Option<&std::sync::Arc<super::event::Recorded>> {
        Some(&self.0)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Waiting on work that failed reports the failure.
#[test]
fn failed_work_errors_on_synchronize() {
    use super::Event;
    let marks = std::sync::Arc::new(super::event::Recorded::default());
    marks.set(0.0, 1.0, false);
    let e = Failed(marks);
    assert!(e.query());
    assert_eq!(
        e.synchronize().unwrap_err(),
        "the work before the event failed on the device"
    );
}

/// Devices without hold support get an empty hold, and dropping it is harmless.
#[test]
fn holds_on_the_cpu_are_empty() {
    let hold = super::hold(Device::Cpu(0)).unwrap();
    assert!(hold.0.is_none());
    drop(hold);
}
