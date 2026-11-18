//! `PinnedAllocator`: page-locked host memory for asynchronous copies between host and CUDA devices. A copy
//! from the device into pageable memory blocks until it finishes; into pinned memory it returns at once.

use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use super::config::PINNED_ALLOCATOR_BYTES;
use super::{Allocator, CpuAllocator, Region};
use crate::device::Device;

/// Host allocator handing out blocks of one `cudaHostAlloc`ed range, reserved on first use; a block that
/// doesn't fit comes from the regular host heap instead (copies into it are synchronous).
pub struct PinnedAllocator {
    region: Region,
}

/// The process's pinned pool, as an allocator for host tensors.
pub(crate) fn pinned() -> Arc<dyn Allocator> {
    Arc::clone(shared()) as Arc<dyn Allocator>
}

/// Whether `ptr` points into the pinned pool's page-locked range.
pub(crate) fn is_pinned(ptr: *const u8) -> bool {
    NonNull::new(ptr.cast_mut()).is_some_and(|ptr| shared().region.contains(ptr))
}

/// The process-wide pinned allocator.
fn shared() -> &'static Arc<PinnedAllocator> {
    static PINNED: OnceLock<Arc<PinnedAllocator>> = OnceLock::new();

    PINNED.get_or_init(|| {
        Arc::new(PinnedAllocator {
            region: Region::new(Device::Cpu(0), PINNED_ALLOCATOR_BYTES),
        })
    })
}

/// Reserves `nbytes` of page-locked host memory; panics when CUDA can't.
fn reserve(nbytes: usize) -> NonNull<u8> {
    let mut ptr = std::ptr::null_mut();
    let err = unsafe { crate::runtime::stream::cuda::ffi::cudaHostAlloc(&mut ptr, nbytes, 0) };

    assert_eq!(
        err, 0,
        "cudaHostAlloc of {nbytes} pinned bytes failed (error {err})"
    );

    NonNull::new(ptr.cast()).expect("cudaHostAlloc returned a pointer")
}

impl Allocator for PinnedAllocator {
    /// A pinned block at `alignment`, or a regular host one when the pool has no room.
    fn allocate(&self, nbytes: usize, alignment: usize) -> NonNull<u8> {
        match self.region.try_take(nbytes, alignment, reserve) {
            Some(ptr) => ptr,
            None => CpuAllocator.allocate(nbytes, alignment),
        }
    }

    /// Returns a block to the pool, or to the host heap when it came from there.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize, alignment: usize) {
        match self.region.contains(ptr) {
            true => self.region.give(ptr, nbytes),
            false => unsafe { CpuAllocator.deallocate(ptr, nbytes, alignment) },
        }
    }
}
