//! The host CPU device backend.

use super::DeviceBackend;

/// The host CPU backend; always one device.
pub struct Cpu;
/// Shared handle to the CPU backend.
pub static CPU: &dyn DeviceBackend = &Cpu;

impl DeviceBackend for Cpu {
    /// Returns `"cpu"`.
    fn name(&self) -> &'static str {
        "cpu"
    }

    /// Always one CPU device.
    fn device_count(&self) -> usize {
        1
    }
}
