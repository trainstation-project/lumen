use super::{Device, DeviceBackend};

#[test]
fn every_device_has_a_backend() {
    for device in [
        Device::Cpu(0),
        Device::Mps(0),
        Device::Cuda(0),
        Device::Meta(0),
    ] {
        assert!(
            device.to_string().starts_with(device.backend().name()),
            "{device}"
        );
    }
    for device in [Device::Cpu(0), Device::Meta(0)] {
        assert_eq!(device.backend().device_count(), 1);
        assert!(device.check().is_ok());
    }
    assert_eq!(
        Device::Mps(0).backend().is_available(),
        Device::Mps(0).check().is_ok()
    );
    let count = super::cuda::CUDA.device_count();
    let err = Device::Cuda(count).check().unwrap_err();
    assert!(
        err.contains(&format!("CUDA device {count} is not available")),
        "{err}"
    );
    assert!(super::cpu::Cpu.check(1).is_err());
}
