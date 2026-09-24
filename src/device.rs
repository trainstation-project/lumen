use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// Apple Silicon GPU via Metal. Single system default device, no index.
    Mps,
    /// Stub for future accelerator support.
    Cuda(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Mps => write!(f, "mps"),
            Device::Cuda(i) => write!(f, "cuda:{i}"),
        }
    }
}
