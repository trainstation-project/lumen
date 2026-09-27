pub mod cache_stats;
pub mod caching;
pub mod config;
mod cpu;
pub mod cuda;
pub mod mps;
pub mod traits;

pub use cpu::CpuAllocator;

use std::alloc::{self, Layout};
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use crate::device::Device;

/// How to release the buffer when the [`DataPtr`] drops.
enum Deleter {
    /// Free with `alloc::dealloc` using the stored layout (CPU path).
    Std,
    /// Custom deletion function to free memory. Used by the CUDA caching
    /// allocator to return the block to its pool rather than freeing it.
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
/// [`caching::CachingAllocator`]) and by raw *backends* alike: to the
/// caching layer, a backend is just an uncached `Allocator` (one
/// `cudaMalloc`/Metal allocation per call).
pub trait Allocator: Send + Sync {
    fn device(&self) -> Device;

    /// Allocate `nbytes`, panicking on failure (c10 semantics: OOM
    /// surfaces as an error, not a return value).
    fn allocate(&self, nbytes: usize) -> DataPtr;

    /// Fallible allocation: `None` when the device cannot satisfy the
    /// request right now.
    ///
    /// Caching allocators use this on their OOM path — release cached
    /// segments, then retry — which must not go through a panic. The
    /// default suits allocators whose failures are unrecoverable (a host
    /// `malloc` failure ends the process anyway): just call
    /// [`allocate`](Self::allocate).
    fn try_allocate(&self, nbytes: usize) -> Option<DataPtr> {
        Some(self.allocate(nbytes))
    }

    /// Copy `nbytes` from host memory at `src` into this allocator's
    /// memory at `dst` (PyTorch: the host-to-device branch of `copy_`).
    ///
    /// The default is a plain `memcpy`, right for host-accessible memory
    /// (CPU, and MPS's shared-storage buffers on unified memory); device
    /// allocators whose memory the host cannot touch (CUDA) override it.
    ///
    /// # Safety
    /// `src` must be valid for reading and `dst` (memory from this
    /// allocator) for writing `nbytes`, and the two must not overlap.
    unsafe fn copy_from_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
    }

    /// Copy `nbytes` from this allocator's memory at `src` into host memory
    /// at `dst` (PyTorch: the device-to-host branch of `copy_`). Same
    /// default and contract as [`copy_from_host`](Self::copy_from_host).
    ///
    /// # Safety
    /// `src` (memory from this allocator) must be valid for reading and
    /// `dst` for writing `nbytes`, and the two must not overlap.
    unsafe fn copy_to_host(&self, dst: *mut u8, src: *const u8, nbytes: usize) {
        unsafe { std::ptr::copy_nonoverlapping(src, dst, nbytes) }
    }
}

/// The allocator for `device` (PyTorch: `c10::GetAllocator`): the CPU
/// allocator, or the device's global caching allocator. Errors if lumen was
/// built without that backend or the device is not present.
pub fn allocator_for(device: Device) -> Result<Arc<dyn Allocator>, String> {
    match device {
        Device::Cpu => {
            static CPU: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
            Ok(Arc::clone(CPU.get_or_init(|| Arc::new(CpuAllocator))))
        }

        Device::Mps => {
            if !mps::is_available() {
                return Err(
                    "MPS is not available (lumen was built without Metal, or there is no \
                     Metal device)"
                        .to_owned(),
                );
            }

            #[cfg(lumen_mps_linked)]
            return Ok(Arc::new(mps::get()));
            #[cfg(not(lumen_mps_linked))]
            unreachable!("mps::is_available() is false without Metal")
        }

        Device::Cuda(index) => {
            let count = cuda::device_count();
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
            unreachable!("cuda::device_count() is 0 without CUDA")
        }
    }
}
