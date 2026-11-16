//! Device allocators: the `Allocator` trait and `allocator_for`, with the owned `DataPtr` buffer
//! (`data_ptr.rs`) and the `Region` sub-allocator accelerator backends carve their tensors from
//! (`region.rs`).

pub mod config;
mod cpu;
pub(crate) mod cuda;
mod data_ptr;
mod meta;
mod mps;
mod region;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;

pub use config::ALIGNMENT;
pub use cpu::CpuAllocator;
#[cfg(lumen_cuda_linked)]
pub use cuda::CudaAllocator;
pub use data_ptr::DataPtr;
pub use meta::MetaAllocator;
#[cfg(lumen_mps_linked)]
pub use mps::MpsAllocator;
#[cfg(lumen_mps_linked)]
pub(crate) use mps::Scratch;
#[cfg(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked, test))]
pub(crate) use region::Region;
#[cfg(lumen_tpu_linked)]
pub use tpu::TpuAllocator;

use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use crate::device::Device;

/// Device memory allocator shared by every buffer on a device.
pub trait Allocator: Send + Sync {
    /// Returns `nbytes` of device memory starting at a multiple of `alignment` (a power of two),
    /// panicking when out of memory.
    fn allocate(&self, nbytes: usize, alignment: usize) -> NonNull<u8>;

    /// Returns `ptr` to the allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must be `nbytes` bytes this allocator allocated at `alignment`, not
    /// yet given back, and unused after the call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize, alignment: usize);
}

/// Returns the shared allocator for `device`, or an error when the device isn't available.
pub fn allocator_for(device: Device) -> Result<Arc<dyn Allocator>, String> {
    device.check()?;

    static CPU: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
    static META: OnceLock<Arc<dyn Allocator>> = OnceLock::new();

    Ok(match device {
        Device::Cpu(_) => Arc::clone(CPU.get_or_init(|| Arc::new(CpuAllocator))),
        Device::Meta(_) => Arc::clone(META.get_or_init(|| Arc::new(MetaAllocator))),
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => mps::allocator(),
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => cuda::allocator(index),
        #[cfg(lumen_tpu_linked)]
        Device::Tpu(index) => tpu::allocator(index),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{device} is never available without its backend"),
    })
}
