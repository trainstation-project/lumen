use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::allocator::{Allocator, DataPtr, allocator_for};
use crate::device::Device;

static NEXT_STORAGE_ID: AtomicUsize = AtomicUsize::new(1);

/// An owning byte buffer on some device (PyTorch: `c10::StorageImpl`).
///
/// The buffer may not be host-addressable (CUDA), so data moves in and out
/// of a device only through the copy ops between tensors
/// ([`crate::ops::copy::copy_h2d`] / [`crate::ops::copy::copy_d2h`]); only
/// CPU storage is dereferenced directly.
pub struct Storage {
    /// Unique id, handy for checking aliasing (PyTorch: object identity of
    /// `StorageImpl`; exposed in python as `tensor.untyped_storage()._cdata`).
    id: usize,
    data: DataPtr,
    nbytes: usize,
    device: Device,
    allocator: Arc<dyn Allocator>,
}

/// The allocator for `device`, panicking if it is unavailable (PyTorch
/// raises on e.g. `torch.zeros(1, device="cuda")` without CUDA).
fn allocator_or_panic(device: Device) -> Arc<dyn Allocator> {
    allocator_for(device).unwrap_or_else(|e| panic!("{e}"))
}

impl Storage {
    /// Allocate `nbytes` of uninitialized memory on `device`, like
    /// PyTorch's `StorageImpl` (the contents are whatever the allocator
    /// returned).
    ///
    /// Sound because nothing public reads raw storage bytes: data comes out
    /// only through a [`crate::Tensor`], which writes before it reads
    /// (`Tensor::zeros` zeroes with the device's memset).
    ///
    /// # Panics
    /// If `device` is not available.
    pub fn new(nbytes: usize, device: Device) -> Self {
        Self::with_allocator(nbytes, allocator_or_panic(device), device)
    }

    /// Allocate `nbytes` of uninitialized memory from `allocator`, which
    /// hands out `device`'s memory (PyTorch: `StorageImpl(size_bytes,
    /// allocator)`), e.g. a custom allocator.
    pub fn with_allocator(nbytes: usize, allocator: Arc<dyn Allocator>, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: allocator.allocate(nbytes),
            nbytes,
            device,
            allocator,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn nbytes(&self) -> usize {
        self.nbytes
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        &self.allocator
    }

    /// A storage over an existing buffer of `nbytes` on `device`, freed by
    /// `data`'s deleter (PyTorch: `at::from_blob`'s storage, e.g. a DLPack
    /// import). `allocator` is the device's, for tensors derived from this
    /// one.
    #[cfg(feature = "python")]
    pub(crate) fn from_data_ptr(
        data: DataPtr,
        nbytes: usize,
        allocator: Arc<dyn Allocator>,
        device: Device,
    ) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data,
            nbytes,
            device,
            allocator,
        }
    }

    /// Raw pointer to the start of the buffer, in the device's address
    /// space: dereferenceable on the host only for host-accessible devices.
    pub fn data_ptr(&self) -> *mut u8 {
        self.data.as_ptr()
    }

    /// Wait for device work in flight before the host touches the buffer:
    /// MPS ops run asynchronously on the MPS stream (PyTorch syncs its
    /// stream before host copies too).
    pub(crate) fn synchronize(&self) {
        #[cfg(lumen_mps_linked)]
        if self.device == Device::Mps {
            crate::stream::mps::synchronize();
        }
    }
}

/// View elements as their bytes. Sound for [`super::dtype::Element`] types, which are
/// plain-old-data without padding.
#[cfg(lumen_mps_linked)]
pub(crate) fn as_bytes<T: super::dtype::Element>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("id", &self.id)
            .field("nbytes", &self.nbytes)
            .field("device", &self.device)
            .finish()
    }
}
