//! `Storage`: a tensor's device allocation with a unique id and parameter state.
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::parameter::Parameter;
use crate::allocator::{Allocator, DataPtr, allocator_for};
use crate::device::Device;

/// Next storage id; ids are never reused.
static NEXT_STORAGE_ID: AtomicUsize = AtomicUsize::new(1);

/// A block of device memory that tensors view.
pub struct Storage {
    id: usize,
    data: DataPtr,
    device: Device,
    /// Parameter state, used when the storage backs a meta parameter.
    pub(crate) parameter: Parameter,
}

/// The allocator for `device`, panicking when the device has none.
fn allocator_or_panic(device: Device) -> Arc<dyn Allocator> {
    allocator_for(device).unwrap_or_else(|e| panic!("{e}"))
}

impl Storage {
    /// Allocates `nbytes` on `device`; panics if the device has no allocator.
    pub fn new(nbytes: usize, device: Device) -> Self {
        Self::with_allocator(nbytes, allocator_or_panic(device), device)
    }

    /// Allocates `nbytes` on `device` aligned to `alignment` bytes.
    pub fn aligned(nbytes: usize, alignment: usize, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: DataPtr::allocate_aligned(allocator_or_panic(device), nbytes, alignment),
            device,
            parameter: Parameter::default(),
        }
    }

    /// Allocates `nbytes` with `allocator`, recording the storage as on `device`.
    pub fn with_allocator(nbytes: usize, allocator: Arc<dyn Allocator>, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: DataPtr::allocate(allocator, nbytes),
            device,
            parameter: Parameter::default(),
        }
    }

    /// Unique id of this allocation.
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn nbytes(&self) -> usize {
        self.data.nbytes()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// The allocator that frees this storage.
    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        self.data.allocator()
    }

    /// Wraps memory allocated elsewhere, such as an adopted DLPack buffer.
    #[cfg(feature = "python")]
    pub(crate) fn from_data_ptr(data: DataPtr, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data,
            device,
            parameter: Parameter::default(),
        }
    }

    /// Address of the first byte.
    pub fn data_ptr(&self) -> *mut u8 {
        self.data.as_ptr()
    }
}

/// The raw bytes of a slice of elements.
pub(crate) fn as_bytes<T: super::dtype::Element>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

/// Shows id, size and device, not the data.
impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("id", &self.id)
            .field("nbytes", &self.nbytes())
            .field("device", &self.device)
            .finish()
    }
}
