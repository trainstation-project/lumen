use super::DeviceBackend;

#[cfg(lumen_mps_linked)]
mod ffi {
    unsafe extern "C" {
        pub fn lumen_mps_available() -> i32;
    }
}

pub struct Mps;
pub static MPS: &dyn DeviceBackend = &Mps;

impl DeviceBackend for Mps {
    fn name(&self) -> &'static str {
        "mps"
    }

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
