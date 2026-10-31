//! Device allocators and the owned `DataPtr` buffer, plus the `Region` sub-allocator
//! accelerator backends carve their tensors from.

pub mod config;
mod cpu;
pub(crate) mod cuda;
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

/// Owned device buffer that returns its bytes to its allocator when dropped.
pub struct DataPtr {
    ptr: NonNull<u8>,
    nbytes: usize,
    allocator: Arc<dyn Allocator>,
}

unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

impl DataPtr {
    /// Allocates `nbytes` from `allocator`.
    pub fn allocate(allocator: Arc<dyn Allocator>, nbytes: usize) -> Self {
        DataPtr {
            ptr: allocator.allocate(nbytes),
            nbytes,
            allocator,
        }
    }

    /// Allocates `nbytes` from `allocator` starting at a multiple of `alignment`.
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

    /// Raw pointer to the buffer's first byte.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Size of the buffer in bytes.
    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    /// Allocator the buffer is returned to.
    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }
}

/// Returns the buffer's bytes to its allocator.
impl Drop for DataPtr {
    fn drop(&mut self) {
        unsafe { self.allocator.deallocate(self.ptr, self.nbytes) }
    }
}

/// Device memory allocator shared by every buffer on a device.
pub trait Allocator: Send + Sync {
    /// Returns `nbytes` of device memory, panicking when out of memory.
    fn allocate(&self, nbytes: usize) -> NonNull<u8>;
    /// Like `allocate` but at a multiple of `alignment`; the default ignores the alignment.
    fn allocate_aligned(&self, nbytes: usize, _alignment: usize) -> NonNull<u8> {
        self.allocate(nbytes)
    }
    /// Returns `ptr` to the allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must be `nbytes` bytes this allocator allocated, not yet
    /// given back, and unused after the call.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize);
}

/// Returns the shared allocator for `device`, or an error when the device isn't available.
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

/// Byte alignment and granularity of every `Region` block.
pub const ALIGNMENT: usize = 256;

/// First-fit sub-allocator over one contiguous device range reserved on first use.
#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
pub(crate) struct Region {
    device: Device,
    /// Bytes reserved for the range.
    capacity: usize,
    state: Mutex<State>,
}

#[cfg_attr(
    not(any(lumen_mps_linked, lumen_cuda_linked, lumen_tpu_linked)),
    allow(dead_code)
)]
/// Bookkeeping of a `Region`; offsets are relative to the reserved base.
struct State {
    /// Start of the reserved range, `None` until the first allocation.
    base: Option<Base>,
    /// Free blocks as offset -> length, coalesced on release.
    free: BTreeMap<usize, usize>,
    /// Allocated blocks as offset -> requested bytes.
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

/// Base pointer of a reserved device range.
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
    /// Region of `capacity` bytes on `device`, rounded down to `ALIGNMENT`, not yet reserved.
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
    /// Bytes the region was created with.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    #[cfg(test)]
    /// End offset of the highest live block, 0 when nothing is live.
    pub(crate) fn used(&self) -> usize {
        self.lock()
            .live
            .last_key_value()
            .map_or(0, |(offset, nbytes)| offset + nbytes)
    }

    #[cfg(test)]
    /// Bytes currently handed out.
    pub(crate) fn live_bytes(&self) -> usize {
        self.lock().live_bytes
    }

    #[cfg_attr(not(any(lumen_mps_linked, lumen_cuda_linked)), allow(dead_code))]
    /// Takes an `ALIGNMENT`-aligned block, see `take_aligned`.
    pub(crate) fn take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
        synchronize: impl Fn(),
    ) -> NonNull<u8> {
        self.take_aligned(nbytes, ALIGNMENT, reserve, synchronize)
    }

    /// Takes `nbytes` at a multiple of `alignment`, reserving the range with `reserve` on first use.
    /// Calls `synchronize` and retries once when no block fits, then panics with the free-space summary.
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
    /// Non-panicking `take` at `ALIGNMENT`.
    pub(crate) fn try_take(
        &self,
        nbytes: usize,
        reserve: impl Fn(usize) -> NonNull<u8>,
    ) -> Option<NonNull<u8>> {
        self.try_take_aligned(nbytes, ALIGNMENT, reserve)
    }

    /// Takes the first free block that fits `nbytes` at `alignment`, or `None` when none does.
    /// A zero-byte request returns the start of the first free block without taking it.
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
            // First fit: the aligned start must leave room for the whole rounded block.
            _ => state.free.iter().find_map(|(&offset, &len)| {
                let start = offset.next_multiple_of(alignment);
                (start + block <= offset + len).then_some((offset, start, len))
            })?,
        };

        let offset = start;
        let base = state.base.as_ref().expect("reserved above").0.as_ptr();
        let ptr = NonNull::new(base.wrapping_add(offset))?;

        // Split the free block into the alignment gap before and the tail after.
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

    /// Returns a block from `take`, merging it with adjacent free blocks.
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
        // Coalesce with the free block ending at `start` and the one starting at its end.
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
    /// Takes the reserved base out so the owning allocator can free it on drop.
    pub(crate) fn base(&mut self) -> Option<NonNull<u8>> {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        state.base.take().map(|base| base.0)
    }

    /// Size of the block holding `nbytes`, rounded up to `ALIGNMENT`.
    fn block(&self, nbytes: usize) -> usize {
        nbytes.next_multiple_of(ALIGNMENT)
    }

    /// Locks the state, ignoring poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}
