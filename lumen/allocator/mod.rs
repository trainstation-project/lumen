pub mod config;
mod cpu;
mod cuda;
mod meta;
mod mps;
#[cfg(test)]
mod tests;
#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;

pub use cpu::CpuAllocator;
#[cfg(lumen_cuda_linked)]
pub use cuda::CudaAllocator;
pub use meta::MetaAllocator;
#[cfg(lumen_mps_linked)]
pub use mps::MpsAllocator;
#[cfg(lumen_mps_linked)]
pub(crate) use mps::Scratch;
#[cfg(lumen_tpu_linked)]
pub use tpu::TpuAllocator;

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

    pub fn allocate_aligned(
        allocator: Arc<dyn Allocator>,
        nbytes: usize,
        alignment: usize,
    ) -> Self {
        DataPtr {
            ptr: allocator.allocate_aligned(nbytes, alignment),
            nbytes,
            allocator,
        }
    }

    /// # Safety
    ///
    /// `ptr` must be `nbytes` bytes `allocator` allocated, owned by the
    /// result alone: it gives them back to `allocator` when dropped.
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
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}

pub trait Allocator: Send + Sync {
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;
    fn allocate_aligned(&self, nbytes: usize, _alignment: usize) -> NonNull<u8> {
        self.allocate(nbytes)
    }
    /// # Safety
    ///
    /// `ptr` must be `nbytes` bytes this allocator allocated, not yet
    /// given back, and unused after the call.
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
        #[cfg(lumen_tpu_linked)]
        Device::Tpu(index) => tpu::allocator(index),
        #[allow(unreachable_patterns)]
        _ => unreachable!("{device} is never available without its backend"),
    })
}

pub const ALIGNMENT: usize = 256;

#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
pub(crate) struct Region {
    device: Device,
    capacity: usize,
    state: Mutex<State>,
}

#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
struct State {
    base: Option<Base>,
    free: BTreeMap<usize, usize>,
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
struct Base(NonNull<u8>);

unsafe impl Send for Base {}

#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
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

    #[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
    pub(crate) fn take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
        synchronize: impl Fn(),
    ) -> NonNull<u8> {
        self.take_aligned(nbytes, ALIGNMENT, reserve, synchronize)
    }

    pub(crate) fn take_aligned(
        &self,
        nbytes: usize,
        alignment: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
        synchronize: impl Fn(),
    ) -> NonNull<u8> {
        let data = self
            .try_take_aligned(nbytes, alignment, &reserve)
            .or_else(|| {
                synchronize();
                self.try_take_aligned(nbytes, alignment, &reserve)
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

    #[cfg(test)]
    pub(crate) fn try_take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
    ) -> Option<NonNull<u8>> {
        self.try_take_aligned(nbytes, ALIGNMENT, reserve)
    }

    pub(crate) fn try_take_aligned(
        &self,
        nbytes: usize,
        alignment: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
    ) -> Option<NonNull<u8>> {
        let mut state = self.lock();
        if state.base.is_none() {
            state.base = Some(Base(reserve(self.capacity)));
        }
        let block = self.block(nbytes);
        let alignment = alignment.max(ALIGNMENT);
        let (free, start, len) = match nbytes {
            0 => {
                let (&offset, &len) = state.free.iter().next().unwrap_or((&0, &0));
                (offset, offset, len)
            }
            _ => state.free.iter().find_map(|(&offset, &len)| {
                let start = offset.next_multiple_of(alignment);
                (start + block <= offset + len).then_some((offset, start, len))
            })?,
        };
        let offset = start;
        let base = state.base.as_ref().expect("reserved above").0.as_ptr();
        let ptr = NonNull::new(base.wrapping_add(offset))?;
        if nbytes > 0 {
            state.free.remove(&free);
            if start > free {
                state.free.insert(free, start - free);
            }
            if free + len > start + block {
                state.free.insert(start + block, free + len - start - block);
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

    #[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
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
