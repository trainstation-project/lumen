#[cfg(lumen_mps_linked)]
use super::static_allocator::StaticAllocator;
#[cfg(lumen_mps_linked)]
use crate::allocator::Allocator;
#[cfg(lumen_mps_linked)]
use crate::device::Device;
#[cfg(lumen_mps_linked)]
use std::ptr::NonNull;
#[cfg(lumen_mps_linked)]
use std::sync::OnceLock;

/// StaticAllocator offsets are aligned to 256 bytes, which Metal's buffer offsets
/// and vector loads are happy with.
pub const ALIGNMENT: usize = 256;

#[cfg(lumen_mps_linked)]
mod ffi {
    // C ABI exported by mps_shim.mm.
    unsafe extern "C" {
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

/// An uncached [`Allocator`] over shared-mode `MTLBuffer`s (via the
/// Objective-C++ shim): one `newBufferWithLength:` per allocation,
/// released by `deallocate`. Only available on macOS
/// builds where the shim was compiled (cfg `lumen_mps_linked`).
#[cfg(lumen_mps_linked)]
pub struct MpsBackend;

#[cfg(lumen_mps_linked)]
impl Allocator for MpsBackend {
    fn allocate(&self, nbytes: usize) -> NonNull<u8> {
        NonNull::new(unsafe { ffi::lumen_mps_alloc(nbytes) })
            .unwrap_or_else(|| panic!("Metal out of memory: failed to allocate {nbytes} bytes"))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, _nbytes: usize) {
        unsafe { ffi::lumen_mps_free(ptr.as_ptr()) }
    }

    /// Submitted MPS work keeps its tensors alive until the GPU has run it.
    fn reclaim(&self) {
        crate::stream::mps::synchronize();
    }

    // Shared buffers are host-addressable: the copy ops reach them with a
    // plain memcpy.
}

/// The MPS device's static allocator.
#[cfg(lumen_mps_linked)]
pub type MpsAllocator = StaticAllocator<MpsBackend>;

/// Get (creating on first use, sized by
/// [`static_allocator_bytes`](super::config::static_allocator_bytes)) the static allocator for the default
/// Metal device.
///
/// Panics if no Metal device is present; check
/// [`crate::device::mps::is_available`] first.
#[cfg(lumen_mps_linked)]
pub fn get() -> MpsAllocator {
    assert!(
        crate::device::mps::is_available(),
        "no Metal device available"
    );
    static ALLOCATOR: OnceLock<MpsAllocator> = OnceLock::new();
    ALLOCATOR
        .get_or_init(|| {
            StaticAllocator::new(
                MpsBackend,
                Device::Mps,
                ALIGNMENT,
                super::config::static_allocator_bytes(),
            )
        })
        .clone()
}
