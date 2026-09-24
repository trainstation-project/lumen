mod cpu;
pub mod cuda;

pub use cpu::CpuAllocator;

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::device::Device;

/// How to release the buffer when the [`DataPtr`] drops.
enum Deleter {
    /// Free with `alloc::dealloc` using the stored layout (CPU path).
    Std,
    /// Custom deleter (PyTorch: `DeleterFnPtr`). Used by the CUDA caching
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
    pub(crate) fn new(ptr: NonNull<u8>, layout: Layout) -> Self {
        DataPtr {
            ptr,
            layout,
            deleter: Deleter::Std,
        }
    }

    /// A `DataPtr` whose drop runs `f` instead of `alloc::dealloc`.
    pub(crate) fn with_deleter(
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

pub trait Allocator: Send + Sync {
    fn device(&self) -> Device;
    fn allocate(&self, nbytes: usize) -> DataPtr;
}
