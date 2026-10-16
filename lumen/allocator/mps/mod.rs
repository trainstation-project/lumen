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

pub const ALIGNMENT: usize = 256;

#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        pub fn lumen_mps_alloc(nbytes: usize) -> *mut u8;
        pub fn lumen_mps_free(ptr: *mut u8);
    }
}

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

    fn reclaim(&self) {
        crate::stream::mps::synchronize();
    }
}

#[cfg(lumen_mps_linked)]
pub type MpsAllocator = StaticAllocator<MpsBackend>;

#[cfg(lumen_mps_linked)]
pub fn get() -> MpsAllocator {
    assert!(
        crate::device::mps::MPS.is_available(),
        "no Metal device available"
    );
    static ALLOCATOR: OnceLock<MpsAllocator> = OnceLock::new();
    ALLOCATOR
        .get_or_init(|| {
            StaticAllocator::new(
                MpsBackend,
                Device::Mps(0),
                ALIGNMENT,
                super::config::static_allocator_bytes(),
            )
        })
        .clone()
}

impl super::AllocatorBackend for crate::device::mps::Mps {
    fn allocator(&self, _index: usize) -> std::sync::Arc<dyn super::Allocator> {
        #[cfg(lumen_mps_linked)]
        return std::sync::Arc::new(get());
        #[cfg(not(lumen_mps_linked))]
        unreachable!("MPS is never available without Metal")
    }
}
