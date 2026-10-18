pub mod cpu;
pub mod cuda;
pub mod meta;
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod tpu;

use std::fmt;

pub trait DeviceBackend: Sync {
    fn name(&self) -> &'static str;
    fn device_count(&self) -> usize;
    fn is_available(&self) -> bool {
        self.device_count() > 0
    }

    fn check(&self, index: usize) -> Result<(), String> {
        let count = self.device_count();
        match index < count {
            true => Ok(()),
            false => Err(format!(
                "{} device {index} is not available ({count} device(s) found)",
                self.name()
            )),
        }
    }

    fn set_device(&self, _index: usize) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu(usize),
    Mps(usize),
    Cuda(usize),
    Tpu(usize),
    Meta(usize),
}

impl Device {
    pub fn of_type(name: &str, index: usize) -> Option<Device> {
        match name {
            "cpu" => Some(Device::Cpu(index)),
            "mps" => Some(Device::Mps(index)),
            "cuda" => Some(Device::Cuda(index)),
            "tpu" => Some(Device::Tpu(index)),
            "meta" => Some(Device::Meta(index)),
            _ => None,
        }
    }

    pub fn backend(self) -> &'static dyn DeviceBackend {
        match self {
            Device::Cpu(_) => cpu::CPU,
            Device::Mps(_) => mps::MPS,
            Device::Cuda(_) => cuda::CUDA,
            Device::Tpu(_) => tpu::TPU,
            Device::Meta(_) => meta::META,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Device::Cpu(index)
            | Device::Mps(index)
            | Device::Cuda(index)
            | Device::Tpu(index)
            | Device::Meta(index) => index,
        }
    }

    pub fn check(self) -> Result<(), String> {
        self.backend().check(self.index())
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend().name(), self.index())
    }
}

#[cfg(test)]
mod tests;
