//! Trainium allocator: one Neuron runtime device tensor per NeuronCore holds every tensor, and each
//! block is also a runtime slice of it, the handle Neuron kernels (NEFFs) take.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use super::{Allocator, Region};
use crate::device::Device;

/// An opaque runtime tensor handle (`nrt_tensor_t *`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Handle(pub(crate) NonNull<c_void>);

unsafe impl Send for Handle {}

/// The Neuron runtime calls the allocator makes: the real runtime when linked, a fake in tests.
pub(crate) trait Runtime: Send + Sync {
    /// Allocates `nbytes` of device memory on NeuronCore `core`: its handle and device address.
    fn allocate(&self, core: usize, nbytes: usize) -> Result<(Handle, NonNull<u8>), String>;
    /// A handle to `nbytes` of `root` starting `offset` bytes in.
    fn slice(&self, root: Handle, offset: usize, nbytes: usize) -> Result<Handle, String>;
    /// Releases a handle from `allocate` or `slice`.
    fn free(&self, tensor: Handle);
}

/// A live block's runtime slice and its size.
struct Block {
    slice: Handle,
    nbytes: usize,
}

/// The root tensor reserved for a device and the slice of each live block, by start address.
#[derive(Default)]
struct Tensors {
    root: Option<(Handle, usize)>,
    blocks: BTreeMap<usize, Block>,
}

/// Allocator for one Trainium device, sub-allocating a single runtime tensor reserved on first use.
pub struct TrainiumAllocator<R: Runtime> {
    index: usize,
    region: Region,
    runtime: R,
    tensors: Mutex<Tensors>,
}

impl<R: Runtime> TrainiumAllocator<R> {
    /// Allocator for device `index` whose region reserves `capacity` bytes through `runtime` on first use.
    pub(crate) fn new(index: usize, capacity: usize, runtime: R) -> Self {
        TrainiumAllocator {
            index,
            region: Region::new(Device::Trainium(index), capacity),
            runtime,
            tensors: Mutex::new(Tensors::default()),
        }
    }

    /// Allocates the root tensor of `nbytes` and returns its device address.
    fn reserve(&self, nbytes: usize) -> NonNull<u8> {
        let (root, base) = self
            .runtime
            .allocate(self.index, nbytes)
            .unwrap_or_else(|e| {
                panic!(
                    "Trainium out of memory: failed to reserve {nbytes} bytes on trainium:{}: {e}",
                    self.index
                )
            });
        self.lock().root = Some((root, base.as_ptr().addr()));
        base
    }

    /// The runtime slice holding `ptr` (the start of a live block or a pointer inside one) and
    /// `ptr`'s byte offset in it, for launching kernels on a tensor or a view of one.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn tensor(&self, ptr: NonNull<u8>) -> Option<(Handle, usize)> {
        let addr = ptr.as_ptr().addr();
        let tensors = self.lock();
        let (&start, block) = tensors.blocks.range(..=addr).next_back()?;
        (addr < start + block.nbytes).then_some((block.slice, addr - start))
    }

    /// Locks the runtime handles, ignoring poisoning.
    fn lock(&self) -> std::sync::MutexGuard<'_, Tensors> {
        self.tensors.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<R: Runtime> Allocator for TrainiumAllocator<R> {
    /// Takes a block of the region and makes its runtime slice; zero-byte blocks get none.
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        let ptr = self.region.take(nbytes, |n| self.reserve(n), || {});
        if nbytes == 0 {
            return ptr;
        }
        let mut tensors = self.lock();
        let (root, base) = tensors.root.expect("the region reserved the root tensor");
        let start = ptr.as_ptr().addr();
        let slice = self
            .runtime
            .slice(root, start - base, nbytes)
            .unwrap_or_else(|e| {
                panic!(
                    "Trainium: failed to slice {nbytes} bytes at {} of trainium:{}'s tensor: {e}",
                    start - base,
                    self.index
                )
            });
        tensors.blocks.insert(start, Block { slice, nbytes });
        ptr
    }

    /// Frees the block's runtime slice and returns the block to the region.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, nbytes: usize) {
        let block = self.lock().blocks.remove(&ptr.as_ptr().addr());
        if let Some(block) = block {
            self.runtime.free(block.slice);
        }
        self.region.give(ptr, nbytes)
    }
}

/// Frees every live slice, then the root tensor if it was ever reserved.
impl<R: Runtime> Drop for TrainiumAllocator<R> {
    fn drop(&mut self) {
        let tensors = std::mem::take(&mut *self.lock());
        for block in tensors.blocks.into_values() {
            self.runtime.free(block.slice);
        }
        if let Some((root, _)) = tensors.root {
            self.runtime.free(root);
        }
    }
}

/// The linked Neuron runtime.
#[cfg(lumen_trainium_linked)]
pub(crate) struct Nrt;

#[cfg(lumen_trainium_linked)]
impl Runtime for Nrt {
    /// `nrt_tensor_allocate`s device memory on the core and reads its address with `nrt_tensor_get_va`.
    fn allocate(&self, core: usize, nbytes: usize) -> Result<(Handle, NonNull<u8>), String> {
        use crate::device::trainium::ffi;
        ffi::init()
            .map_err(|status| format!("the Neuron runtime failed to start: status {status}"))?;
        let mut tensor = std::ptr::null_mut();
        let status = unsafe {
            ffi::nrt_tensor_allocate(
                ffi::NRT_TENSOR_PLACEMENT_DEVICE,
                core as i32,
                nbytes,
                c"lumen".as_ptr(),
                &mut tensor,
            )
        };
        let handle = NonNull::new(tensor)
            .filter(|_| status == ffi::NRT_SUCCESS)
            .ok_or_else(|| format!("nrt_tensor_allocate failed: status {status}"))?;
        let base = NonNull::new(unsafe { ffi::nrt_tensor_get_va(handle.as_ptr()) }.cast());
        match base {
            Some(base) => Ok((Handle(handle), base)),
            None => {
                self.free(Handle(handle));
                Err("nrt_tensor_get_va returned null".into())
            }
        }
    }

    /// `nrt_tensor_allocate_slice` of the root tensor.
    fn slice(&self, root: Handle, offset: usize, nbytes: usize) -> Result<Handle, String> {
        use crate::device::trainium::ffi;
        let mut slice = std::ptr::null_mut();
        let status = unsafe {
            ffi::nrt_tensor_allocate_slice(
                root.0.as_ptr(),
                offset,
                nbytes,
                c"lumen".as_ptr(),
                &mut slice,
            )
        };
        NonNull::new(slice)
            .filter(|_| status == ffi::NRT_SUCCESS)
            .map(Handle)
            .ok_or_else(|| format!("nrt_tensor_allocate_slice failed: status {status}"))
    }

    /// `nrt_tensor_free`.
    fn free(&self, tensor: Handle) {
        let mut tensor = tensor.0.as_ptr();
        unsafe { crate::device::trainium::ffi::nrt_tensor_free(&mut tensor) };
    }
}

/// Returns the shared allocator for Trainium device `index`, creating it on first use.
#[cfg(lumen_trainium_linked)]
pub(super) fn allocator(index: usize) -> std::sync::Arc<dyn Allocator> {
    use std::collections::HashMap;
    use std::sync::{Arc, OnceLock};
    static ALLOCATORS: OnceLock<Mutex<HashMap<usize, Arc<dyn Allocator>>>> = OnceLock::new();
    let mut allocators = ALLOCATORS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let capacity = super::config::static_allocator_bytes();
    Arc::clone(
        allocators
            .entry(index)
            .or_insert_with(|| Arc::new(TrainiumAllocator::new(index, capacity, Nrt))),
    )
}
