use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::parameter::Parameter;
use crate::allocator::{Allocator, DataPtr, allocator_for};
use crate::device::Device;

static NEXT_STORAGE_ID: AtomicUsize = AtomicUsize::new(1);

pub struct Storage {
    id: usize,
    data: DataPtr,
    device: Device,
    pub(crate) parameter: Parameter,
}

fn allocator_or_panic(device: Device) -> Arc<dyn Allocator> {
    allocator_for(device).unwrap_or_else(|e| panic!("{e}"))
}

impl Storage {
    pub fn new(nbytes: usize, device: Device) -> Self {
        Self::with_allocator(nbytes, allocator_or_panic(device), device)
    }

    pub fn aligned(nbytes: usize, alignment: usize, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: DataPtr::allocate_aligned(allocator_or_panic(device), nbytes, alignment),
            device,
            parameter: Parameter::default(),
        }
    }

    pub fn with_allocator(nbytes: usize, allocator: Arc<dyn Allocator>, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: DataPtr::allocate(allocator, nbytes),
            device,
            parameter: Parameter::default(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn nbytes(&self) -> usize {
        self.data.nbytes()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn allocator(&self) -> &Arc<dyn Allocator> {
        self.data.allocator()
    }

    #[cfg(feature = "python")]
    pub(crate) fn from_data_ptr(data: DataPtr, device: Device) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data,
            device,
            parameter: Parameter::default(),
        }
    }

    pub fn data_ptr(&self) -> *mut u8 {
        self.data.as_ptr()
    }
}

#[cfg(lumen_mps_linked)]
pub(crate) fn as_bytes<T: super::dtype::Element>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("id", &self.id)
            .field("nbytes", &self.nbytes())
            .field("device", &self.device)
            .finish()
    }
}
