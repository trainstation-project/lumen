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

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use crate::device::Device;

/// How to release the buffer when the [`DataPtr`] drops.
enum Deleter {
    /// Free with `alloc::dealloc` using the stored layout (CPU path).
    Std,
    /// Custom deletion function to free memory: a backend's release, or an
    /// static allocator's bookkeeping.
    Custom(Box<dyn FnOnce(NonNull<u8>) + Send + Sync>),
}

/// An owning, type-erased data pointer with an associated deleter.
pub struct DataPtr {
    ptr: NonNull<u8>,
    layout: Layout,
    deleter: Deleter,
}

// The buffer is plain bytes; sending it to another thread is fine as long
// as access is synchronized, which `Arc<Storage>` provides for sharing.
unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    /// A `DataPtr` freed with `alloc::dealloc` on drop (CPU path).
    pub fn new(ptr: NonNull<u8>, layout: Layout) -> Self {
        DataPtr {
            ptr,
            layout,
            deleter: Deleter::Std,
        }
    }

    /// A `DataPtr` whose drop runs `f` instead of `alloc::dealloc`.
    ///
    /// `ptr` must point to `layout.size()` bytes that stay valid until the
    /// `DataPtr` is dropped; `f` must release (or recycle) that buffer.
    pub fn with_deleter(
        ptr: NonNull<u8>,
        layout: Layout,
        f: impl FnOnce(NonNull<u8>) + Send + Sync + 'static,
    ) -> Self {
        DataPtr {
            ptr,
            layout,
            deleter: Deleter::Custom(Box::new(f)),
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for DataPtr {
    fn drop(&mut self) {
        match std::mem::replace(&mut self.deleter, Deleter::Std) {
            Deleter::Std => {
                if self.layout.size() > 0 {
                    unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
                }
            }

            Deleter::Custom(f) => f(self.ptr),
        }
    }
}

/// The memory-management contract: allocate `nbytes` on a device, get back
/// an owning [`DataPtr`] that releases the memory on drop.
///
/// Implemented by user-facing allocators ([`cpu::CpuAllocator`],
/// [`static_allocator::StaticAllocator`]) and by raw *backends* alike: to an allocator, a backend is
/// just an uncached `Allocator` (one `cudaMalloc`/Metal allocation per
/// call), asked once for the static allocator's region.
pub trait Allocator: Send + Sync {
    /// Allocate `nbytes`, panicking on failure (c10 semantics: OOM
    /// surfaces as an error, not a return value).
    fn allocate(&self, nbytes: usize) -> DataPtr;

    /// Fallible allocation: `None` when the device cannot satisfy the
    /// request right now (a static allocator reserving its region uses this). The
    /// default suits allocators whose failures are unrecoverable (a host
    /// `malloc` failure ends the process anyway): just call
    /// [`allocate`](Self::allocate).
    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        Some(self.allocate(nbytes))
    }

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
