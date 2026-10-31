//! CUDA allocator: hands out blocks of one `cudaMalloc`ed static region per device.

#[cfg(lumen_cuda_linked)]
use std::collections::HashMap;
#[cfg(lumen_cuda_linked)]
use std::ptr::NonNull;
#[cfg(lumen_cuda_linked)]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(lumen_cuda_linked)]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(lumen_cuda_linked)]
use super::{Allocator, Region};
#[cfg(lumen_cuda_linked)]
use crate::device::Device;

#[cfg(lumen_cuda_linked)]
use crate::device::cuda::CUDA;

/// Raw CUDA runtime allocation entry points.
#[cfg(lumen_cuda_linked)]
mod ffi {
    use std::ffi::c_void;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
        pub fn cudaFree(devPtr: *mut c_void) -> i32;
    }
}

/// `cudaMalloc`s `nbytes` on device `index`, panicking when it fails.
#[cfg(lumen_cuda_linked)]
fn malloc(index: usize, nbytes: usize) -> NonNull<u8> {
    CUDA.set_device(index);
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let err = unsafe { ffi::cudaMalloc(&mut ptr, nbytes) };

    NonNull::new(ptr.cast())
        .filter(|_| err == 0)
        .unwrap_or_else(|| {
            panic!("CUDA out of memory: failed to allocate {nbytes} bytes on cuda:{index}")
        })
}

/// Frees a `malloc`ed device pointer on device `index`.
///
/// # Safety
/// `ptr` is a `malloc` of device `index`'s, not yet freed.
#[cfg(lumen_cuda_linked)]
unsafe fn free(index: usize, ptr: NonNull<u8>) {
    CUDA.set_device(index);

    unsafe { ffi::cudaFree(ptr.as_ptr().cast()) };
}

/// Allocator for one CUDA device, sub-allocating a lazily reserved static region.
#[cfg(lumen_cuda_linked)]
pub struct CudaAllocator {
    index: usize,
    region: Region,
}

#[cfg(lumen_cuda_linked)]
impl CudaAllocator {
    /// Allocator for device `index` whose region reserves `capacity` bytes on first use.
    pub fn new(index: usize, capacity: usize) -> Self {
        CudaAllocator {
            index,
            region: Region::new(Device::Cuda(index), capacity),
        }
    }
}

#[cfg(lumen_cuda_linked)]
impl Allocator for CudaAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let ptr = self.region.take(nbytes, |n| malloc(self.index, n), || {});

        if GRAPHS.load(Ordering::Acquire) > 0
            && let Some(held) = holds().get_mut(&self.index)
            && held.capturing
        {
            held.blocks.insert(ptr.as_ptr() as usize, None);
            held.taken.push(ptr.as_ptr() as usize);
        }

        ptr
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        // A block a CUDA graph launches into is given back when the graph goes, not when freed.
        if GRAPHS.load(Ordering::Acquire) > 0
            && let Some(held) = holds().get_mut(&self.index)
            && let Some(freed) = held.blocks.get_mut(&(ptr.as_ptr() as usize))
        {
            *freed = Some(nbytes);

            return;
        }

        self.region.give(ptr, nbytes)
    }
}

/// Frees the device region if it was ever reserved.
#[cfg(lumen_cuda_linked)]
impl Drop for CudaAllocator {
    fn drop(&mut self) {
        if let Some(base) = self.region.base() {
            unsafe { free(self.index, base) }
        }
    }
}

/// A device's blocks taken while a CUDA graph captured, each with its size once freed: the graph
/// launches into them, so a freed one returns to the region only when its graph is dropped.
#[cfg(lumen_cuda_linked)]
#[derive(Default)]
struct Held {
    capturing: bool,
    taken: Vec<usize>,
    blocks: HashMap<usize, Option<usize>>,
}

/// CUDA graphs capturing or alive, which hold blocks; while none is, freeing skips `holds`.
#[cfg(lumen_cuda_linked)]
static GRAPHS: AtomicUsize = AtomicUsize::new(0);

/// Each device's held blocks.
#[cfg(lumen_cuda_linked)]
fn holds() -> std::sync::MutexGuard<'static, HashMap<usize, Held>> {
    static HOLDS: OnceLock<Mutex<HashMap<usize, Held>>> = OnceLock::new();

    HOLDS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Starts holding the blocks device `index` hands out, for a CUDA graph capturing.
#[cfg(lumen_cuda_linked)]
pub(crate) fn hold(index: usize) {
    GRAPHS.fetch_add(1, Ordering::AcqRel);
    holds().entry(index).or_default().capturing = true;
}

/// Stops holding new blocks on device `index`; returns those taken since `hold`, which stay held.
#[cfg(lumen_cuda_linked)]
pub(crate) fn held(index: usize) -> Vec<usize> {
    let mut holds = holds();
    let held = holds.entry(index).or_default();

    held.capturing = false;
    std::mem::take(&mut held.taken)
}

/// Lets go of `blocks` of device `index`: those already freed return to the region, the others
/// when freed.
#[cfg(lumen_cuda_linked)]
pub(crate) fn release(index: usize, blocks: &[usize]) {
    let freed: Vec<(usize, usize)> = {
        let mut holds = holds();
        let held = holds.entry(index).or_default();

        blocks
            .iter()
            .filter_map(|b| held.blocks.remove(b).flatten().map(|n| (*b, n)))
            .collect()
    };

    let allocator = allocator(index);

    for (block, nbytes) in freed {
        let ptr = NonNull::new(block as *mut u8).expect("a held block is not null");

        unsafe { allocator.deallocate(ptr, nbytes) };
    }

    GRAPHS.fetch_sub(1, Ordering::AcqRel);
}

/// Returns the shared allocator for CUDA device `index`, creating it on first use.
#[cfg(lumen_cuda_linked)]
pub(super) fn allocator(index: usize) -> Arc<dyn Allocator> {
    static ALLOCATORS: OnceLock<Mutex<HashMap<usize, Arc<dyn Allocator>>>> = OnceLock::new();

    let mut allocators = ALLOCATORS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let capacity = super::config::static_allocator_bytes();

    Arc::clone(
        allocators
            .entry(index)
            .or_insert_with(|| Arc::new(CudaAllocator::new(index, capacity))),
    )
}
