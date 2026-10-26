//! The CUDA device backend, querying the CUDA runtime when lumen is linked against it.

use super::DeviceBackend;

/// CUDA runtime and driver entry points used for device discovery and selection.
#[cfg(lumen_cuda_linked)]
mod ffi {
    use std::ffi::c_void;

    #[link(name = "cudart")]
    unsafe extern "C" {
        pub fn cudaGetDeviceCount(count: *mut i32) -> i32;
        pub fn cudaGetDevice(device: *mut i32) -> i32;
        pub fn cudaSetDevice(device: i32) -> i32;
    }

    #[link(name = "cuda")]
    unsafe extern "C" {
        pub fn cuCtxGetCurrent(ctx: *mut *mut c_void) -> i32;
    }
}

/// The CUDA backend; reports no devices when built without CUDA.
pub struct Cuda;
/// Shared handle to the CUDA backend.
pub static CUDA: &dyn DeviceBackend = &Cuda;

impl DeviceBackend for Cuda {
    /// Returns `"cuda"`.
    fn name(&self) -> &'static str {
        "cuda"
    }

    /// Number of visible CUDA devices, or 0 if the runtime call fails or CUDA is not linked.
    fn device_count(&self) -> usize {
        #[cfg(lumen_cuda_linked)]
        {
            let mut count = 0;
            match unsafe { ffi::cudaGetDeviceCount(&mut count) } {
                0 => count.max(0) as usize,
                _ => 0,
            }
        }
        #[cfg(not(lumen_cuda_linked))]
        {
            0
        }
    }

    /// Like the default check, but mentions when lumen was built without CUDA.
    fn check(&self, index: usize) -> Result<(), String> {
        let count = self.device_count();
        if index < count {
            return Ok(());
        }
        let missing = match cfg!(lumen_cuda_linked) {
            true => "",
            false => "; lumen was built without CUDA",
        };
        Err(format!(
            "CUDA device {index} is not available ({count} device(s) found{missing})"
        ))
    }

    /// Makes `index` the current CUDA device unless it already is and has a live context.
    #[cfg(lumen_cuda_linked)]
    fn set_device(&self, index: usize) {
        let mut current = -1;
        let mut context = std::ptr::null_mut();
        unsafe {
            if ffi::cudaGetDevice(&mut current) != 0
                || current != index as i32
                || ffi::cuCtxGetCurrent(&mut context) != 0
                || context.is_null()
            {
                ffi::cudaSetDevice(index as i32);
            }
        }
    }
}
