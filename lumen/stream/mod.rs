//! Device streams: a handle to lumen's work queue on each device for host synchronization,
//! and holds that stall a queue.

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

/// A device's queue of lumen work.
pub trait Stream: Send + Sync {
    /// Blocks the host until all queued work finishes.
    fn synchronize(&self);
}

/// lumen's stream on `device`; errors if the device is not available.
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

/// Keeps a device's queue stalled until dropped; holds nothing on devices without hold support.
pub(crate) struct Hold(Option<Box<dyn FnOnce() + Send>>);

impl Drop for Hold {
    /// Releases the stalled queue.
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            release();
        }
    }
}

/// Stalls `device`'s queue so work enqueued meanwhile does not start until the returned
/// `Hold` drops; only CUDA and MPS queues are stalled, other devices get an empty hold.
pub(crate) fn hold(device: Device) -> Result<Hold, String> {
    device.check()?;

    Ok(Hold(match device {
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => Some(cuda::hold(index)?),
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => Some(mps::hold()?),
        _ => None,
    }))
}
