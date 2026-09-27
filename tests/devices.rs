//! Device selection for tensors and storage, on any machine.

use lumen::allocator::{allocator_for, cuda, mps};
use lumen::{Device, Storage, Tensor};

#[test]
fn cpu_is_always_available() {
    assert_eq!(allocator_for(Device::Cpu).unwrap().device(), Device::Cpu);
    let t = Tensor::zeros_on::<f32>(&[2], Device::Cpu);
    assert_eq!(t.device(), Device::Cpu);
    assert_eq!(Storage::new(8, Device::Cpu).device(), Device::Cpu);
}

#[test]
fn storage_new_is_zeroed() {
    let s = Storage::new(16, Device::Cpu);
    let mut bytes = [0xFFu8; 16];
    s.read_bytes(0, &mut bytes);
    assert_eq!(bytes, [0; 16]);
}

#[test]
fn unavailable_devices_are_reported() {
    if !cuda::is_available() {
        let err = allocator_for(Device::Cuda(0)).err().unwrap();
        assert!(err.contains("CUDA device 0 is not available"), "{err}");
    }
    let past_last = Device::Cuda(cuda::device_count());
    assert!(allocator_for(past_last).is_err());
    if !mps::is_available() {
        assert!(allocator_for(Device::Mps).is_err());
    }
}

#[test]
#[should_panic(expected = "is not available")]
fn tensor_on_unavailable_device_panics() {
    Tensor::zeros_on::<f32>(&[1], Device::Cuda(cuda::device_count()));
}
