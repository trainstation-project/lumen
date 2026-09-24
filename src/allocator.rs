//! Allocator abstraction, modeled on `c10::Allocator`.
//!
//! An allocator hands out [`DataPtr`]s: an owning raw pointer plus the
//! layout needed to free it (PyTorch carries a deleter fn in its
//! `UniqueVoidPtr`; Rust lets us use `alloc::Layout` + `Drop` instead).

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::device::Device;

/// An owning, type-erased data pointer with an associated deleter.
/// Mirrors `c10::DataPtr`.
pub struct DataPtr {
    ptr: NonNull<u8>,
    layout: Layout,
}

// The buffer is plain bytes; sending it to another thread is fine as long
// as access is synchronized, which `Arc<Storage>` provides for sharing.
unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for DataPtr {
    fn drop(&mut self) {
        if self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}

pub trait Allocator: Send + Sync {
    fn device(&self) -> Device;
    fn allocate(&self, nbytes: usize) -> DataPtr;
}

/// CPU allocator backed by the global allocator (c10: `CPUAllocator`).
pub struct CpuAllocator;

static CPU_ALLOCATOR: CpuAllocator = CpuAllocator;

impl CpuAllocator {
    pub fn get() -> &'static dyn Allocator {
        &CPU_ALLOCATOR
    }
}

impl Allocator for CpuAllocator {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn allocate(&self, nbytes: usize) -> DataPtr {
        // 64-byte alignment, like c10::gAlignment, to keep SIMD loads happy.
        let layout = Layout::from_size_align(nbytes, 64).expect("invalid allocation layout");

        let ptr = if nbytes == 0 {
            // Layout with size 0 is fine; dangling but aligned.
            NonNull::new(layout.align() as *mut u8).unwrap()
        } else {
            let raw = unsafe { alloc::alloc_zeroed(layout) };
            NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        };

        DataPtr { ptr, layout }
    }
}
