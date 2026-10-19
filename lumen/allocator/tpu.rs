//! The TPU allocator: a [`Region`] over one buffer of each device's memory,
//! reserved through libtpu's PJRT API (`lumen/device/tpu`) on the first
//! allocation. A TPU's memory is not addressable from the host: a pointer
//! this allocator hands out is the runtime's address of the region's buffer
//! plus the allocation's offset in it, never dereferenced; [`copy_to_host`]
//! reads an allocation through the buffer at that offset, and [`run`] runs
//! a program on the buffer (the only way to write it).
//!
//! The buffer is TPU memory in rows of 512 bytes (`u32[rows, 1, 128]`),
//! which a program addresses by row: every allocation is whole rows.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use super::{Allocator, Region};
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

/// The bytes of a row of the region: what a program addresses it in.
pub(crate) const ROW: usize = 512;

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

    /// The region's memory: the `nbytes` it allocates in (whole rows).
    fn reserve(&self, nbytes: usize) -> NonNull<u8> {
        let nbytes = nbytes - nbytes % ROW;
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

    /// Runs `executable` on the region (see `lumen_tpu_run`), its scalars
    /// the rows of `rows` (allocations of this allocator) and then `scalars`,
    /// and `host` (whole rows) copied to the TPU for it.
    fn run(
        &self,
        executable: *mut c_void,
        rows: &[*const u8],
        scalars: &[i32],
        host: &[u8],
    ) -> Result<(), String> {
        let mut reserved = self.lock();
        let Some(Reserved { buffer, base, size }) = reserved.as_mut() else {
            return Err(format!("nothing is allocated on tpu:{}", self.index));
        };
        let row = |ptr: &*const u8| {
            let offset = ptr.addr().wrapping_sub(*base);
            match offset < *size && offset % ROW == 0 {
                true => Ok((offset / ROW) as i32),
                false => Err(format!(
                    "TPU programs read and write whole rows of {ROW} bytes: offset {offset} of \
                     tpu:{}'s memory is not one",
                    self.index
                )),
            }
        };
        let rows = rows.iter().map(row).collect::<Result<Vec<_>, _>>()?;
        let scalars = [rows.as_slice(), scalars].concat();
        let mut region = buffer.as_ptr();
        // SAFETY: a compiled program, the region's buffer (replaced by the
        // run's result, the same memory) and readable scalars and bytes.
        let failed = unsafe {
            ffi::lumen_tpu_run(
                executable,
                self.index as i32,
                scalars.as_ptr(),
                scalars.len(),
                host.as_ptr().cast(),
                host.len(),
                &mut region,
            )
        };
        *buffer = NonNull::new(region).expect("a run's result is the region");
        match failed {
            0 => Ok(()),
            _ => Err(error()),
        }
    }

    /// Waits for the programs run on the region so far.
    fn wait(&self) -> Result<(), String> {
        let reserved = self.lock();
        let Some(Reserved { buffer, .. }) = reserved.as_ref() else {
            return Ok(());
        };
        // SAFETY: the region's buffer.
        match unsafe { ffi::lumen_tpu_wait(buffer.as_ptr()) } {
            0 => Ok(()),
            _ => Err(error()),
        }
    }

    /// The region's rows: a program's view of it is `u32[rows, 1, 128]`.
    fn rows(&self) -> usize {
        self.lock().as_ref().map_or(0, |r| r.size / ROW)
    }
}

impl Allocator for TpuAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let nbytes = nbytes.next_multiple_of(ROW);
        self.region.take(nbytes, |n| self.reserve(n), || {})
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        self.region.give(ptr, nbytes.next_multiple_of(ROW))
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

/// Device `index`'s allocator, if it has allocated.
fn allocated(index: usize) -> Result<Arc<TpuAllocator>, String> {
    let allocator = allocators().get(&index).cloned();
    allocator.ok_or_else(|| format!("nothing is allocated on tpu:{index}"))
}

/// Runs `executable` on device `index`'s memory: see [`TpuAllocator::run`].
pub(crate) fn run(
    index: usize,
    executable: *mut c_void,
    rows: &[*const u8],
    scalars: &[i32],
    host: &[u8],
) -> Result<(), String> {
    allocated(index)?.run(executable, rows, scalars, host)
}

/// Waits for the programs run on device `index`'s memory so far (none if
/// nothing is allocated there).
pub(crate) fn wait(index: usize) -> Result<(), String> {
    let allocator = allocators().get(&index).cloned();
    allocator.map_or(Ok(()), |a| a.wait())
}

/// The rows of device `index`'s memory, as a program sees it.
pub(crate) fn rows(index: usize) -> Result<usize, String> {
    Ok(allocated(index)?.rows())
}

/// Copies `dst.len()` bytes of TPU memory at `ptr` (an allocation of
/// device `index`'s allocator, at least that long) to `dst`, waiting for
/// them.
pub(crate) fn copy_to_host(index: usize, ptr: NonNull<u8>, dst: &mut [u8]) -> Result<(), String> {
    allocated(index)?.read(ptr, dst)
}
