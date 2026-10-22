mod cpu;
#[cfg(lumen_cuda_linked)]
pub mod cuda;
pub mod event;
mod meta;
#[cfg(lumen_mps_linked)]
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
mod tpu;

pub use cpu::CpuStream;
#[cfg(lumen_cuda_linked)]
pub use cuda::CudaStream;
pub use event::{Event, event_for};
pub use meta::MetaStream;
#[cfg(lumen_mps_linked)]
pub use mps::MpsStream;
#[cfg(lumen_tpu_linked)]
pub use tpu::TpuStream;

use std::sync::Arc;

use crate::device::Device;

pub trait Stream: Send + Sync {
    fn synchronize(&self);
}

pub fn stream_for(device: Device) -> Result<Arc<dyn Stream>, String> {
    device.check()?;
    Ok(match device {
        Device::Cpu(_) => Arc::new(CpuStream),
        Device::Meta(_) => Arc::new(MetaStream),
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => Arc::new(MpsStream),
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => Arc::new(CudaStream { index }),
        #[cfg(lumen_tpu_linked)]
        Device::Tpu(index) => Arc::new(TpuStream { index }),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{device} is never available without its backend"),
    })
}
