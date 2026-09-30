#[cfg(lumen_mps_linked)]
mod ffi {
    // C ABI exported by allocator/mps_shim.mm.
    unsafe extern "C" {
        pub fn lumen_mps_available() -> i32;
    }
}

/// True when lumen was built with Metal support (macOS + `mps` feature) and
/// a default Metal device exists. Safe to call on any build.
pub fn is_available() -> bool {
    #[cfg(lumen_mps_linked)]
    {
        unsafe { ffi::lumen_mps_available() != 0 }
    }
    #[cfg(not(lumen_mps_linked))]
    {
        false
    }
}
