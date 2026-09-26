use std::alloc::{self, Layout};
use std::ptr::NonNull;

use super::{Allocator, DataPtr};
use crate::device::Device;

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
        // 64-byte alignment to keep SIMD loads happy.
        let layout = Layout::from_size_align(nbytes, 64).expect("invalid allocation layout");

        let ptr = if nbytes == 0 {
            // Layout with size 0 is fine; dangling but aligned.
            NonNull::new(std::ptr::without_provenance_mut(layout.align())).unwrap()
        } else {
            // Uninitialized memory, like C's `malloc`. Reading
            // uninitialized memory is undefined behaviour in Rust.
            let raw = unsafe { alloc::alloc(layout) };
            NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        };

        DataPtr::new(ptr, layout)
    }
}
