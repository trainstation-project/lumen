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
pub use meta::MetaAllocator;

use std::collections::{BTreeMap, HashMap};
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
        Device::Mps(_) | Device::Cuda(_) => shared(device),
    })
}

pub const ALIGNMENT: usize = 256;

fn shared(device: Device) -> Arc<dyn Allocator> {
    static ALLOCATORS: OnceLock<Mutex<HashMap<Device, Arc<dyn Allocator>>>> = OnceLock::new();
    let mut allocators = ALLOCATORS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let capacity = config::static_allocator_bytes();
    Arc::clone(
        allocators
            .entry(device)
            .or_insert_with(|| Arc::new(StaticAllocator::new(device, capacity))),
    )
}

// The device's own memory, the region reserved from it (the CPU's: the
// system allocator's, for tests).
fn malloc(device: Device, nbytes: usize) -> NonNull<u8> {
    match device {
        Device::Cpu(_) => CpuAllocator.allocate(nbytes),
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => mps::malloc(nbytes),
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => cuda::malloc(index, nbytes),
        _ => unreachable!("{device} has no memory to reserve"),
    }
}

/// # Safety
/// `ptr` and `nbytes` are a `malloc` of `device`'s, not yet freed.
unsafe fn free(device: Device, ptr: NonNull<u8>, nbytes: usize) {
    match device {
        Device::Cpu(_) => unsafe { CpuAllocator.deallocate(ptr, nbytes) },
        #[cfg(lumen_mps_linked)]
        Device::Mps(_) => unsafe { mps::free(ptr) },
        #[cfg(lumen_cuda_linked)]
        Device::Cuda(index) => unsafe { cuda::free(index, ptr) },
        _ => unreachable!("{device} has no memory to free"),
    }
}

// Wait for device work in flight, which keeps freed memory alive (MPS work
// holds its tensors until the GPU has run it).
fn synchronize(device: Device) {
    #[cfg(lumen_mps_linked)]
    if let Device::Mps(_) = device {
        crate::stream::mps::synchronize();
    }
    let _ = device;
}

pub struct StaticAllocator {
    device: Device,
    capacity: usize,
    state: Mutex<State>,
}

struct State {
    /// The region, reserved on the first allocation.
    region: Option<Region>,
    /// Free blocks, offset to length; offsets and lengths are multiples of
    /// the alignment, and no two blocks touch (they are merged).
    free: BTreeMap<usize, usize>,
    /// Allocations not yet freed, offset to requested bytes, and the sum of
    /// their bytes.
    live: BTreeMap<usize, usize>,
    live_bytes: usize,
}

/// The start of the region, in the backend's address space.
struct Region(NonNull<u8>);

// SAFETY: the region is device memory owned by the allocator; the pointer
// is only offset and handed out, never dereferenced here.
unsafe impl Send for Region {}

impl StaticAllocator {
    pub fn new(device: Device, capacity: usize) -> Self {
        StaticAllocator {
            device,
            capacity,
            state: Mutex::new(State {
                region: None,
                free: BTreeMap::from([(0, capacity - capacity % ALIGNMENT)]),
                live: BTreeMap::new(),
                live_bytes: 0,
            }),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn used(&self) -> usize {
        self.lock()
            .live
            .last_key_value()
            .map_or(0, |(offset, nbytes)| offset + nbytes)
    }

    pub fn live_bytes(&self) -> usize {
        self.lock().live_bytes
    }

    fn block(&self, nbytes: usize) -> usize {
        nbytes.next_multiple_of(ALIGNMENT)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Return the allocation of `nbytes` at `offset` (address `addr`),
    /// merging its block with free neighbours.
    fn free(&self, addr: usize, offset: usize, nbytes: usize) {
        let mut state = self.lock();
        if nbytes == 0 {
            return;
        }
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

    /// An allocation of `nbytes`, or `None` when the region has no room
    /// left. Reserves the region on first use (panicking if the backend
    /// cannot).
    pub(crate) fn try_allocate(&self, nbytes: usize) -> Option<NonNull<u8>> {
        let mut state = self.lock();
        if state.region.is_none() {
            state.region = Some(Region(malloc(self.device, self.capacity)));
        }
        let block = self.block(nbytes);
        // An empty allocation takes no block: any aligned offset will do.
        let (&offset, &len) = match nbytes {
            0 => state.free.iter().next().unwrap_or((&0, &0)),
            _ => state.free.iter().find(|&(_, &len)| len >= block)?,
        };
        let base = state.region.as_ref().expect("reserved above").0.as_ptr();
        // In the device's address space: the backend's memory is not a Rust
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
}

impl Allocator for StaticAllocator {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        // Out of room: memory freed but held by device work in flight comes
        // back once that work is done.
        let data = self.try_allocate(nbytes).or_else(|| {
            synchronize(self.device);
            self.try_allocate(nbytes)
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

    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        let base = match &self.lock().region {
            Some(region) => region.0.as_ptr().addr(),
            None => unreachable!("an allocation reserves the region first"),
        };
        let addr = ptr.as_ptr().addr();
        self.free(addr, addr - base, nbytes);
    }
}

impl Drop for StaticAllocator {
    fn drop(&mut self) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(region) = state.region.take() {
            // SAFETY: the device allocated the region with this size, and
            // nothing points into it any more.
            unsafe { free(self.device, region.0, self.capacity) };
        }
    }
}
