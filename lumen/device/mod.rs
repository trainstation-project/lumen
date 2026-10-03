pub mod cpu;
pub mod cuda;
pub mod mps;

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Mps,
    Cuda(usize),
    /// No memory (PyTorch: the meta device): a tensor here has a shape,
    /// dtype and strides but no data. Tracing, shape inference and
    /// checkpoint headers use it; reading its data is an error.
    Meta,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
            Device::Meta => write!(f, "meta"),
        }
    }
}
