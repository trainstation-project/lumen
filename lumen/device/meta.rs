use super::DeviceBackend;

pub struct Meta;
pub static META: &dyn DeviceBackend = &Meta;

impl DeviceBackend for Meta {
    fn name(&self) -> &'static str {
        "meta"
    }

    fn device_count(&self) -> usize {
        1
    }
}
