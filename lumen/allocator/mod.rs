pub mod config;
mod cpu;
pub mod cuda;
pub mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
pub mod static_allocator;
#[cfg(test)]
mod tests;

pub use cpu::CpuAllocator;

/// The meta device's allocator: no memory. Its pointers are dangling and
/// never dereferenced (meta tensors have no data to read or write).
pub struct MetaAllocator;

impl Allocator for MetaAllocator {
    fn allocate(&self, _nbytes: usize) -> NonNull<u8> {
        NonNull::dangling()
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _nbytes: usize) {}
}

use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use crate::device::Device;

/// An owned allocation (PyTorch: `c10::DataPtr`): `nbytes` at a pointer
/// from `allocator`, given back to it when the `DataPtr` drops.
pub struct DataPtr {
    ptr: NonNull<u8>,
    nbytes: usize,
    allocator: Arc<dyn Allocator>,
}

// The buffer is plain bytes; sending it to another thread is fine as long
// as access is synchronized, which `Arc<Storage>` provides for sharing.
unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    /// `nbytes` from `allocator`.
    pub fn allocate(allocator: Arc<dyn Allocator>, nbytes: usize) -> Self {
        DataPtr {
            ptr: allocator.allocate(nbytes),
            nbytes,
            allocator,
        }
    }

    /// Own an existing allocation, which `allocator` releases on drop.
    ///
    /// # Safety
    /// `ptr` and `nbytes` must be an allocation `allocator` can release
    /// (see [`Allocator::deallocate`]), released by nothing else.
    pub unsafe fn from_raw_parts(
        ptr: NonNull<u8>,
        nbytes: usize,
        allocator: Arc<dyn Allocator>,
    ) -> Self {
        DataPtr {
            ptr,
            nbytes,
            allocator,
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }
}

impl Drop for DataPtr {
    fn drop(&mut self) {
        // SAFETY: an allocation of `allocator`'s, released only here.
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}

/// The memory-management contract: allocate `nbytes` on a device, and
/// release them again (PyTorch: c10's `raw_allocate` / `raw_deallocate`).
/// A [`DataPtr`] owns one allocation and gives it back when it drops.
///
/// Implemented by user-facing allocators ([`cpu::CpuAllocator`],
/// [`static_allocator::StaticAllocator`]) and by raw *backends* alike: to an allocator, a backend is
/// just an uncached `Allocator` (one `cudaMalloc`/Metal allocation per
/// call), asked once for the static allocator's region.
pub trait Allocator: Send + Sync {
    /// Allocate `nbytes`, panicking on failure (c10 semantics: OOM
    /// surfaces as an error, not a return value).
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;

    /// Release an allocation.
    ///
    /// # Safety
    /// `ptr` and `nbytes` are an allocation this allocator returned, not yet
    /// released, which nothing uses any more.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize);

    /// Wait for device work in flight that keeps freed memory alive, so it
    /// can be reused: a static allocator out of room calls this, then tries
    /// again. MPS work holds its tensors until the GPU has run it; the
    /// default suits devices whose frees take effect at once.
    fn reclaim(&self) {}
}

/// The allocator for `device` (PyTorch: `c10::GetAllocator`): the CPU
/// allocator, or the device's allocator. Errors if lumen was built without that
/// backend or the device is not present.
pub fn allocator_for(device: Device) -> Result<Arc<dyn Allocator>, String> {
    match device {
        Device::Cpu => {
            static CPU: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
            Ok(Arc::clone(CPU.get_or_init(|| Arc::new(CpuAllocator))))
        }

        Device::Mps => {
            if !crate::device::mps::is_available() {
                return Err(
                    "MPS is not available (lumen was built without Metal, or there is no \
                     Metal device)"
                        .to_owned(),
                );
            }

            #[cfg(lumen_mps_linked)]
            return Ok(Arc::new(mps::get()));
            #[cfg(not(lumen_mps_linked))]
            unreachable!("device::mps::is_available() is false without Metal")
        }

        Device::Meta => {
            static META: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
            Ok(Arc::clone(META.get_or_init(|| Arc::new(MetaAllocator))))
        }

        Device::Cuda(index) => {
            let count = crate::device::cuda::device_count();
            if index >= count {
                return Err(format!(
                    "CUDA device {index} is not available ({count} device(s) found{})",
                    if cfg!(lumen_cuda_linked) {
                        ""
                    } else {
                        "; lumen was built without CUDA"
                    }
                ));
            }
            #[cfg(lumen_cuda_linked)]
            return Ok(Arc::new(cuda::get(index)));
            #[cfg(not(lumen_cuda_linked))]
            unreachable!("device::cuda::device_count() is 0 without CUDA")
        }
    }
}
