//! The MPS (Apple Metal) device backend.

use super::DeviceBackend;

/// Metal availability probe from the Objective-C runtime shim.
#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        pub fn lumen_mps_available() -> i32;
    }
}

/// The Metal backend: at most one device.
pub struct Mps;
/// Shared handle to the MPS backend.
pub static MPS: &dyn DeviceBackend = &Mps;

impl DeviceBackend for Mps {
    /// Returns `"mps"`.
    fn name(&self) -> &'static str {
        "mps"
    }

    /// One if a Metal device is present and lumen was built with Metal, else zero.
    fn device_count(&self) -> usize {
        #[cfg(lumen_mps_linked)]
        {
            usize::from(unsafe { ffi::lumen_mps_available() != 0 })
        }
        #[cfg(not(lumen_mps_linked))]
        {
            0
        }
    }

    /// Accepts only index 0, with a message saying why Metal is missing.
    fn check(&self, index: usize) -> Result<(), String> {
        match (self.is_available(), index) {
            (true, 0) => Ok(()),
            (true, _) => Err(format!(
                "MPS device {index} is not available (1 device(s) found)"
            )),
            (false, _) => Err(
                "MPS is not available (lumen was built without Metal, or there is no \
                 Metal device)"
                    .to_owned(),
            ),
        }
    }
}
