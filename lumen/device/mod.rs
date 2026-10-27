//! Devices (`cpu`, `mps`, `cuda`, `tpu`, `trainium`, `meta`) and the backend trait each implements.

pub mod cpu;
pub mod cuda;
pub mod meta;
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod tpu;
pub mod trainium;

use std::fmt;

/// Per-device-type queries: name, how many devices exist, and how to select one.
pub trait DeviceBackend: Sync {
    /// Lowercase device type name, as used in `"name:index"`.
    fn name(&self) -> &'static str;
    /// Number of devices of this type present on this machine.
    fn device_count(&self) -> usize;
    /// Whether at least one device of this type exists.
    fn is_available(&self) -> bool {
        self.device_count() > 0
    }

    /// `Ok` if device `index` exists, else a message saying why not.
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

    /// Makes `index` the current device before launching work on it; a no-op by default.
    fn set_device(&self, _index: usize) {}
}

/// A device type and index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu(usize),
    Mps(usize),
    Cuda(usize),
    Tpu(usize),
    Trainium(usize),
    Meta(usize),
}

impl Device {
    /// The device of type `name` (`"cpu"`, `"mps"`, ...) at `index`, or `None` for an unknown type.
    pub fn of_type(name: &str, index: usize) -> Option<Device> {
        match name {
            "cpu" => Some(Device::Cpu(index)),
            "mps" => Some(Device::Mps(index)),
            "cuda" => Some(Device::Cuda(index)),
            "tpu" => Some(Device::Tpu(index)),
            "trainium" => Some(Device::Trainium(index)),
            "meta" => Some(Device::Meta(index)),
            _ => None,
        }
    }

    /// The backend for this device's type.
    pub fn backend(self) -> &'static dyn DeviceBackend {
        match self {
            Device::Cpu(_) => cpu::CPU,
            Device::Mps(_) => mps::MPS,
            Device::Cuda(_) => cuda::CUDA,
            Device::Tpu(_) => tpu::TPU,
            Device::Trainium(_) => trainium::TRAINIUM,
            Device::Meta(_) => meta::META,
        }
    }

    /// The device's index within its type.
    pub fn index(self) -> usize {
        match self {
            Device::Cpu(index)
            | Device::Mps(index)
            | Device::Cuda(index)
            | Device::Tpu(index)
            | Device::Trainium(index)
            | Device::Meta(index) => index,
        }
    }

    /// `Ok` if this device exists on this machine.
    pub fn check(self) -> Result<(), String> {
        self.backend().check(self.index())
    }
}

impl fmt::Display for Device {
    /// Formats as `"type:index"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend().name(), self.index())
    }
}

#[cfg(test)]
mod tests;
