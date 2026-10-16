use super::DeviceBackend;

pub struct Cpu;
pub static CPU: &dyn DeviceBackend = &Cpu;

impl DeviceBackend for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_count(&self) -> usize {
        1
    }
}
