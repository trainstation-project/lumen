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
