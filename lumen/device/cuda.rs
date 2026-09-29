//! CUDA devices (PyTorch: `torch.cuda`'s device functions): how many there
//! are, and which is current. Without a CUDA toolkit at build time
//! (`build.rs`), there are none.

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

/// Number of visible CUDA devices (`torch.cuda.device_count()`); 0 when
/// lumen was built without CUDA or the driver reports an error.
pub fn device_count() -> usize {
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

/// Whether any CUDA device is usable (`torch.cuda.is_available()`).
pub fn is_available() -> bool {
    device_count() > 0
}

/// Make `device_index` the current device, skipping the call if it already
/// is (PyTorch: `c10::cuda::SetDevice`). On a fresh thread `cudaGetDevice`
/// reports device 0 with no context current, which the driver API calls
/// launched there need, so the call is skipped only once one is.
#[cfg(lumen_cuda_linked)]
pub(crate) fn set_device(device_index: usize) {
    let mut current = -1;
    let mut context = std::ptr::null_mut();
    unsafe {
        if ffi::cudaGetDevice(&mut current) != 0
            || current != device_index as i32
            || ffi::cuCtxGetCurrent(&mut context) != 0
            || context.is_null()
        {
            ffi::cudaSetDevice(device_index as i32);
        }
    }
}
