use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::dtype::Element;
use crate::allocator::{Allocator, DataPtr, allocator_for};
use crate::device::Device;

static NEXT_STORAGE_ID: AtomicUsize = AtomicUsize::new(1);

/// An owning byte buffer on some device (PyTorch: `c10::StorageImpl`).
///
/// The buffer may not be host-addressable (CUDA), so data moves in and out
/// only through the copy ops ([`crate::ops::copy::copy_h2d`] /
/// [`crate::ops::copy::copy_d2h`]), never by dereferencing the pointer.
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
        Self::with_allocator(nbytes, allocator_or_panic(device))
    }

    /// Allocate `nbytes` of uninitialized memory from `allocator` (PyTorch:
    /// `StorageImpl(size_bytes, allocator)`), e.g. a custom allocator.
    pub fn with_allocator(nbytes: usize, allocator: Arc<dyn Allocator>) -> Self {
        Storage {
            id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
            data: allocator.allocate(nbytes),
            nbytes,
            device: allocator.device(),
            allocator,
        }
    }

    /// A storage on `device` holding a copy of `data` (PyTorch:
    /// `torch.tensor(data, device=...)`).
    ///
    /// # Panics
    /// If `device` is not available.
    pub fn from_slice<T: Element>(data: &[T], device: Device) -> Self {
        let storage = Self::new(size_of_val(data), device);
        // Fills every byte of the fresh buffer.
        storage.write(0, data);
        storage
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

    /// Raw pointer to the start of the buffer, in the device's address
    /// space: dereferenceable on the host only for host-accessible devices.
    pub fn data_ptr(&self) -> *mut u8 {
        self.data.as_ptr()
    }

    /// Copy `out.len()` bytes starting at byte `offset` to the host.
    ///
    /// Crate-private: the storage may be uninitialized (see
    /// [`new`](Self::new)), so only the tensor layer, which tracks what has
    /// been written, reads it.
    ///
    /// # Panics
    /// If the range is out of bounds.
    pub(crate) fn read_bytes(&self, offset: usize, out: &mut [u8]) {
        self.check_range(offset, out.len());
        if out.is_empty() {
            return;
        }
        self.synchronize();
        // SAFETY: the range is in bounds of this storage's buffer, and `out`
        // is a distinct host buffer of the same length.
        unsafe {
            crate::ops::copy::copy_d2h(
                self.device,
                out.as_mut_ptr(),
                self.ptr_at(offset),
                out.len(),
            );
        }
    }

    /// Copy `bytes` from the host into the buffer at byte `offset`.
    ///
    /// Crate-private: a storage has no dtype of its own, so arbitrary bytes
    /// could later be read as an invalid value (e.g. a `bool` that is
    /// neither 0 nor 1); the tensor layer only writes values of the
    /// tensor's dtype. Like [`crate::Tensor::set`], this writes shared data
    /// through a shared reference; the caller must ensure no data races.
    ///
    /// # Panics
    /// If the range is out of bounds.
    pub(crate) fn write_bytes(&self, offset: usize, bytes: &[u8]) {
        self.check_range(offset, bytes.len());
        if bytes.is_empty() {
            return;
        }
        self.synchronize();
        // SAFETY: as in `read_bytes`.
        unsafe {
            crate::ops::copy::copy_h2d(
                self.device,
                self.ptr_at(offset),
                bytes.as_ptr(),
                bytes.len(),
            );
        }
    }

    /// Copy `len` elements of type `T`, starting at element `offset`, to
    /// the host. Crate-private: the bytes must have been written as `T`.
    pub(crate) fn read<T: Element>(&self, offset: usize, len: usize) -> Vec<T> {
        let mut out = vec![T::ZERO; len];
        self.read_bytes(offset * size_of::<T>(), as_bytes_mut(&mut out));
        out
    }

    /// Copy `data` into the buffer starting at element `offset`. Same
    /// caveats as [`write_bytes`](Self::write_bytes).
    pub(crate) fn write<T: Element>(&self, offset: usize, data: &[T]) {
        self.write_bytes(offset * size_of::<T>(), as_bytes(data));
    }

    /// Wait for device work in flight before the host touches the buffer:
    /// MPS ops run asynchronously on the MPS stream (PyTorch syncs its
    /// stream before host copies too).
    fn synchronize(&self) {
        #[cfg(lumen_mps_linked)]
        if self.device == Device::Mps {
            crate::stream::mps::synchronize();
        }
    }

    /// The buffer's address `offset` bytes in, in the device's address
    /// space. `wrapping_add`, not `add`: device memory (CUDA, or a mock
    /// device's fake addresses) is not an allocation Rust knows about, and
    /// `add` is undefined behavior outside one. The allocator's copies and
    /// memset are what access it, never a Rust dereference.
    fn ptr_at(&self, offset: usize) -> *mut u8 {
        self.data.as_ptr().wrapping_add(offset)
    }

    fn check_range(&self, offset: usize, len: usize) {
        assert!(
            offset
                .checked_add(len)
                .is_some_and(|end| end <= self.nbytes),
            "byte range {offset}..{} out of bounds for storage of {} bytes",
            offset.saturating_add(len),
            self.nbytes
        );
    }
}

/// View elements as their bytes. Sound for [`Element`] types, which are
/// plain-old-data without padding.
pub(crate) fn as_bytes<T: Element>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
}

/// Mutable byte view of initialized elements. Callers only store bytes
/// that were read back from values of the same type `T`, so every element
/// stays a valid `T` (this matters for `bool`).
fn as_bytes_mut<T: Element>(data: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), size_of_val(data)) }
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
