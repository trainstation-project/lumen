pub mod config;
mod cpu;
mod cuda;
mod meta;
mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

pub use cpu::CpuAllocator;
#[cfg(lumen_cuda_linked)]
pub use cuda::CudaAllocator;
pub use meta::MetaAllocator;
#[cfg(lumen_mps_linked)]
pub use mps::MpsAllocator;

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use crate::device::Device;

pub struct DataPtr {
    ptr: NonNull<u8>,
    nbytes: usize,
    allocator: Arc<dyn Allocator>,
}

unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    pub fn allocate(allocator: Arc<dyn Allocator>, nbytes: usize) -> Self {
        DataPtr {
            ptr: allocator.allocate(nbytes),
            nbytes,
            allocator,
        }
    }

    pub unsafe fn from_raw_parts(
        ptr: NonNull<u8>,
        nbytes: usize,
        allocator: Arc<dyn Allocator>,
    ) -> Self {
        DataPtr {
            ptr,
            nbytes,
            allocator,
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }
}

impl Drop for DataPtr {
    fn drop(&mut self) {
        // SAFETY: an allocation of `allocator`'s, released only here.
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}

pub trait Allocator: Send + Sync {
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize);
}

pub fn allocator_for(device: Device) -> Result<Arc<dyn Allocator>, String> {
    device.check()?;
    static CPU: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
    static META: OnceLock<Arc<dyn Allocator>> = OnceLock::new();
    Ok(match device {
        Device::Cpu(_) => Arc::clone(CPU.get_or_init(|| Arc::new(CpuAllocator))),
        Device::Meta(_) => Arc::clone(META.get_or_init(|| Arc::new(MetaAllocator))),
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => mps::allocator(),
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => cuda::allocator(index),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{device} is never available without its backend"),
    })
}

pub const ALIGNMENT: usize = 256;

/// A range of a device's memory (MPS's, CUDA's), reserved on first use and
/// handed out first fit, blocks merged when freed. Not an allocator: the
/// device's allocator owns it, reserving and releasing the range.
pub(crate) struct Region {
    device: Device,
    capacity: usize,
    state: Mutex<State>,
}

struct State {
    /// The range's start, reserved on the first allocation.
    base: Option<Base>,
    /// Free blocks, offset to length; offsets and lengths are multiples of
    /// the alignment, and no two blocks touch (they are merged).
    free: BTreeMap<usize, usize>,
    /// Allocations not yet freed, offset to requested bytes, and the sum of
    /// their bytes.
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

struct Base(NonNull<u8>);

// SAFETY: the range is device memory owned by the allocator; the pointer
// is only offset and handed out, never dereferenced here.
unsafe impl Send for Base {}

impl Region {
    pub(crate) fn new(device: Device, capacity: usize) -> Self {
        Region {
            device,
            capacity,
            state: Mutex::new(State {
                base: None,
                free: BTreeMap::from([(0, capacity - capacity % ALIGNMENT)]),
                live: BTreeMap::new(),
                live_bytes: 0,
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> usize {
        self.lock()
            .live
            .last_key_value()
            .map_or(0, |(offset, nbytes)| offset + nbytes)
    }

    #[cfg(test)]
    pub(crate) fn live_bytes(&self) -> usize {
        self.lock().live_bytes
    }

    pub(crate) fn take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
        synchronize: impl Fn(),
    ) -> NonNull<u8> {
        let data = self.try_take(nbytes, &reserve).or_else(|| {
            synchronize();
            self.try_take(nbytes, &reserve)
        });
        data.unwrap_or_else(|| {
            let state = self.lock();
            panic!(
                "{} static allocator out of memory: {nbytes} bytes requested, {} of {} bytes free \
                 (largest block {}); raise {}.config.static_allocator_bytes",
                self.device,
                state.free.values().sum::<usize>(),
                self.capacity,
                state.free.values().max().unwrap_or(&0),
                crate::LIBRARY_NAME,
            )
        })
    }

    pub(crate) fn try_take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
    ) -> Option<NonNull<u8>> {
        let mut state = self.lock();
        if state.base.is_none() {
            state.base = Some(Base(reserve(self.capacity)));
        }
        let block = self.block(nbytes);
        // An empty allocation takes no block: any aligned offset will do.
        let (&offset, &len) = match nbytes {
            0 => state.free.iter().next().unwrap_or((&0, &0)),
            _ => state.free.iter().find(|&(_, &len)| len >= block)?,
        };
        let base = state.base.as_ref().expect("reserved above").0.as_ptr();
        // In the device's address space: its memory is not a Rust
        // allocation, so no `add`.
        let ptr = NonNull::new(base.wrapping_add(offset))?;
        if nbytes > 0 {
            state.free.remove(&offset);
            if len > block {
                state.free.insert(offset + block, len - block);
            }
            state.live.insert(offset, nbytes);
            state.live_bytes += nbytes;
        }
        let live_bytes = state.live_bytes;
        drop(state);
        let addr = ptr.as_ptr().addr();
        crate::profiler::report_memory(self.device, addr, nbytes as i64, live_bytes, self.capacity);
        Some(ptr)
    }

    pub(crate) fn give(&self, ptr: NonNull<u8>, nbytes: usize) {
        let mut state = self.lock();
        let base = match &state.base {
            Some(base) => base.0.as_ptr().addr(),
            None => unreachable!("an allocation reserves the range first"),
        };
        if nbytes == 0 {
            return;
        }
        let addr = ptr.as_ptr().addr();
        let offset = addr - base;
        state.live.remove(&offset);
        state.live_bytes -= nbytes;
        let (mut start, mut len) = (offset, self.block(nbytes));
        if let Some((&prev, &prev_len)) = state.free.range(..start).next_back()
            && prev + prev_len == start
        {
            state.free.remove(&prev);
            (start, len) = (prev, prev_len + len);
        }
        if let Some(next_len) = state.free.remove(&(start + len)) {
            len += next_len;
        }
        state.free.insert(start, len);
        let live_bytes = state.live_bytes;
        drop(state);
        crate::profiler::report_memory(
            self.device,
            addr,
            -(nbytes as i64),
            live_bytes,
            self.capacity,
        );
    }

    pub(crate) fn base(&mut self) -> Option<NonNull<u8>> {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        state.base.take().map(|base| base.0)
    }

    fn block(&self, nbytes: usize) -> usize {
        nbytes.next_multiple_of(ALIGNMENT)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}
