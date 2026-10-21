use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use super::{Allocator, Region};
use crate::device::Device;
use crate::device::tpu::{error, ffi};

struct Reserved {
    buffer: NonNull<c_void>,
    base: usize,
    size: usize,
}

unsafe impl Send for Reserved {}

pub(crate) const ROW: usize = 512;

const TILE_ROWS: usize = 8 * 64 * 9 * 5 * 7;

pub struct TpuAllocator {
    index: usize,
    region: Region,
    reserved: Mutex<Option<Reserved>>,
}

impl TpuAllocator {
    pub fn new(index: usize, capacity: usize) -> Self {
        let unit = TILE_ROWS * ROW;
        let capacity = match capacity >= unit {
            true => capacity - capacity % unit,
            false => capacity,
        };
        TpuAllocator {
            index,
            region: Region::new(Device::Tpu(index), capacity),
            reserved: Mutex::new(None),
        }
    }

    fn reserve(&self, nbytes: usize) -> NonNull<u8> {
        let nbytes = nbytes - nbytes % ROW;
        let mut base = 0;
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

    fn rows_of(&self, ptrs: &[*const u8]) -> Result<Vec<i32>, String> {
        let reserved = self.lock();
        let Some(&Reserved { base, size, .. }) = reserved.as_ref() else {
            return Err(format!("nothing is allocated on tpu:{}", self.index));
        };
        let row = |ptr: &*const u8| {
            let offset = ptr.addr().wrapping_sub(base);
            match offset < size && offset % ROW == 0 {
                true => Ok((offset / ROW) as i32),
                false => Err(format!(
                    "TPU programs read and write whole rows of {ROW} bytes: offset {offset} of \
                     tpu:{}'s memory is not one",
                    self.index
                )),
            }
        };
        ptrs.iter().map(row).collect()
    }

    fn run(&self, executable: *mut c_void, scalars: &[i32], host: &[u8]) -> Result<(), String> {
        let mut reserved = self.lock();
        let Some(Reserved { buffer, .. }) = reserved.as_mut() else {
            return Err(format!("nothing is allocated on tpu:{}", self.index));
        };
        let mut region = buffer.as_ptr();
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

    fn wait(&self) -> Result<(), String> {
        let reserved = self.lock();
        let Some(Reserved { buffer, .. }) = reserved.as_ref() else {
            return Ok(());
        };
        match unsafe { ffi::lumen_tpu_wait(buffer.as_ptr()) } {
            0 => Ok(()),
            _ => Err(error()),
        }
    }

    fn rows(&self) -> usize {
        self.lock().as_ref().map_or(0, |r| r.size / ROW)
    }
}

impl Allocator for TpuAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        self.allocate_aligned(nbytes, ROW)
    }

    fn allocate_aligned(&self, nbytes: usize, alignment: usize) -> NonNull<u8> {
        let nbytes = nbytes.next_multiple_of(ROW);
        self.region
            .take_aligned(nbytes, alignment.max(ROW), |n| self.reserve(n), || {})
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        self.region.give(ptr, nbytes.next_multiple_of(ROW))
    }
}

impl Drop for TpuAllocator {
    fn drop(&mut self) {
        if let Some(Reserved { buffer, .. }) = self.lock().take() {
            unsafe { ffi::lumen_tpu_free(buffer.as_ptr()) }
        }
    }
}

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

fn allocated(index: usize) -> Result<Arc<TpuAllocator>, String> {
    let allocator = allocators().get(&index).cloned();
    allocator.ok_or_else(|| format!("nothing is allocated on tpu:{index}"))
}

pub(crate) fn run(
    index: usize,
    executable: *mut c_void,
    scalars: &[i32],
    host: &[u8],
) -> Result<(), String> {
    allocated(index)?.run(executable, scalars, host)
}

pub(crate) fn rows_of(index: usize, ptrs: &[*const u8]) -> Result<Vec<i32>, String> {
    allocated(index)?.rows_of(ptrs)
}

pub(crate) fn wait(index: usize) -> Result<(), String> {
    let allocator = allocators().get(&index).cloned();
    allocator.map_or(Ok(()), |a| a.wait())
}

pub(crate) fn rows(index: usize) -> Result<usize, String> {
    Ok(allocated(index)?.rows())
}

pub(crate) fn copy_to_host(index: usize, ptr: NonNull<u8>, dst: &mut [u8]) -> Result<(), String> {
    allocated(index)?.read(ptr, dst)
}
