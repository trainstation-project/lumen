//! The meta device backend: tensors with types and shapes but no data.

use super::DeviceBackend;

/// The meta backend; always one device.
pub struct Meta;
/// Shared handle to the meta backend.
pub static META: &dyn DeviceBackend = &Meta;

impl DeviceBackend for Meta {
    /// Returns `"meta"`.
    fn name(&self) -> &'static str {
        "meta"
    }

    /// Always one meta device.
    fn device_count(&self) -> usize {
        1
    }
}
