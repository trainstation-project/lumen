//! The TPU allocator: a [`Region`] over one buffer of each device's memory,
//! reserved through libtpu's PJRT API (`lumen/device/tpu`) on the first
//! allocation. A TPU's memory is not addressable from the host: a pointer
//! this allocator hands out is the runtime's address of the region's buffer
//! plus the allocation's offset in it, never dereferenced; [`copy_to_host`]
//! reads an allocation through the buffer at that offset.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use super::{ALIGNMENT, Allocator, Region};
use crate::device::Device;
use crate::device::tpu::{error, ffi};

/// The region's PJRT buffer, its address (the allocations' base) and its
/// bytes.
struct Reserved {
    buffer: NonNull<c_void>,
    base: usize,
    size: usize,
}

// SAFETY: a PJRT buffer handle, which PJRT's API takes from any thread.
unsafe impl Send for Reserved {}

pub struct TpuAllocator {
    index: usize,
    region: Region,
    reserved: Mutex<Option<Reserved>>,
}

impl TpuAllocator {
    pub fn new(index: usize, capacity: usize) -> Self {
        TpuAllocator {
            index,
            region: Region::new(Device::Tpu(index), capacity),
            reserved: Mutex::new(None),
        }
    }

    /// The region's memory: the `nbytes` it allocates in (whole blocks of
    /// the alignment, so whole u32s: the region's buffer's elements).
    fn reserve(&self, nbytes: usize) -> NonNull<u8> {
        let nbytes = nbytes - nbytes % ALIGNMENT;
        let mut base = 0;
        // SAFETY: a writable address.
        let buffer = unsafe { ffi::lumen_tpu_alloc(self.index as i32, nbytes, &mut base) };
        let reserved = NonNull::new(buffer).zip(NonNull::new(base as *mut u8));
        let (buffer, ptr) = reserved.unwrap_or_else(|| {
            panic!(
                "TPU out of memory: failed to reserve {nbytes} bytes on tpu:{}: {}",
                self.index,
                error()
            )
        });
        *self.lock() = Some(Reserved {
            buffer,
            base,
            size: nbytes,
        });
        ptr
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<Reserved>> {
        self.reserved.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `dst.len()` bytes of the allocation at `ptr` (one of this
    /// allocator's, at least that long) to `dst`, waiting for them.
    fn read(&self, ptr: NonNull<u8>, dst: &mut [u8]) -> Result<(), String> {
        let reserved = self.lock();
        let Some(&Reserved { buffer, base, size }) = reserved.as_ref() else {
            return Err(format!("nothing is allocated on tpu:{}", self.index));
        };
        let offset = ptr.as_ptr().addr().wrapping_sub(base);
        if offset.checked_add(dst.len()).is_none_or(|end| end > size) {
            return Err(format!(
                "{} bytes at offset {offset} are outside tpu:{}'s {size}",
                dst.len(),
                self.index
            ));
        }
        // SAFETY: the region's buffer, `size` bytes, read inside them into
        // `dst`'s writable bytes.
        let failed = unsafe {
            ffi::lumen_tpu_read(
                buffer.as_ptr(),
                size,
                offset,
                dst.as_mut_ptr().cast(),
                dst.len(),
            )
        };
        match failed {
            0 => Ok(()),
            _ => Err(error()),
        }
    }
}

impl Allocator for TpuAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        self.region.take(nbytes, |n| self.reserve(n), || {})
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        self.region.give(ptr, nbytes)
    }
}

impl Drop for TpuAllocator {
    fn drop(&mut self) {
        if let Some(Reserved { buffer, .. }) = self.lock().take() {
            // SAFETY: the region's buffer, which nothing points into any more.
            unsafe { ffi::lumen_tpu_free(buffer.as_ptr()) }
        }
    }
}

/// Each device's allocator, created on its first use.
fn allocators() -> std::sync::MutexGuard<'static, HashMap<usize, Arc<TpuAllocator>>> {
    static ALLOCATORS: OnceLock<Mutex<HashMap<usize, Arc<TpuAllocator>>>> = OnceLock::new();
    ALLOCATORS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub(super) fn allocator(index: usize) -> Arc<dyn Allocator> {
    let capacity = super::config::static_allocator_bytes();
    let mut allocators = allocators();
    let allocator = allocators
        .entry(index)
        .or_insert_with(|| Arc::new(TpuAllocator::new(index, capacity)));
    Arc::clone(allocator) as Arc<dyn Allocator>
}

/// Copies `dst.len()` bytes of TPU memory at `ptr` (an allocation of
/// device `index`'s allocator, at least that long) to `dst`, waiting for
/// them.
pub(crate) fn copy_to_host(index: usize, ptr: NonNull<u8>, dst: &mut [u8]) -> Result<(), String> {
    let allocator = allocators().get(&index).cloned();
    match allocator {
        Some(allocator) => allocator.read(ptr, dst),
        None => Err(format!("nothing is allocated on tpu:{index}")),
    }
}
