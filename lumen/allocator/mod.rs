pub mod cache_stats;
pub mod caching;
mod cpu;
pub mod cuda;
pub mod mps;
pub mod traits;

pub use cpu::CpuAllocator;

use std::alloc::{self, Layout};
use std::ptr::NonNull;

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
}
