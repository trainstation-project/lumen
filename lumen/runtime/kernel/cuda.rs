//! Launching a CUDA kernel.

use super::Launch;
use crate::codegen::utils::cuda::Cuda;
use std::ffi::{CStr, c_char, c_uint, c_void};

use crate::runtime::triton::direct::{Compiled, Value, bytes, driver};

/// One CUDA launch: a compiled kernel, its grid and its runtime parameters in order.
pub(crate) struct CudaLaunch<'a> {
    pub(crate) kernel: &'a Compiled,
    pub(crate) grid: [usize; 3],
    pub(crate) values: &'a [Value],
}

/// CUDA enqueues a kernel on lumen's stream with `cuLaunchKernel`, straight from Rust.
impl Launch for Cuda {
    type Args<'a> = CudaLaunch<'a>;

    /// Launches its `kernel` on `grid` with `values` (its runtime parameters in order) on lumen's stream (a
    /// CUDA graph's capture stream while it captures), then the two scratch pointers Triton appends,
    /// null since the kernel needs no scratch.
    fn launch(args: CudaLaunch<'_>) -> Result<(), String> {
        let CudaLaunch {
            kernel,
            grid,
            values,
        } = args;

        if kernel.types.len() != values.len() {
            return Err(format!(
                "a Triton kernel of {} runtime parameters launched with {} values",
                kernel.types.len(),
                values.len()
            ));
        }

        let mut storage: Vec<[u8; 8]> = Vec::with_capacity(values.len() + 2);

        for (ty, &value) in kernel.types.iter().zip(values) {
            if let Some(b) = bytes(ty, value)? {
                storage.push(b);
            }
        }

        storage.extend([[0u8; 8]; 2]);
        let mut params: Vec<*mut c_void> =
            storage.iter_mut().map(|b| b.as_mut_ptr().cast()).collect();

        if grid.contains(&0) {
            return Ok(());
        }

        let driver = driver()?;
        let [x, y, z] = grid.map(|g| g as c_uint);

        let result = unsafe {
            (driver.launch)(
                kernel.function as *mut c_void,
                x,
                y,
                z,
                32 * kernel.warps,
                1,
                1,
                kernel.shared,
                crate::runtime::stream::cuda::launch_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };

        match result {
            0 => Ok(()),
            e => {
                let mut message: *const c_char = std::ptr::null();

                unsafe { (driver.error)(e, &mut message) };

                let message = match message.is_null() {
                    true => "unknown".into(),
                    false => unsafe { CStr::from_ptr(message) }.to_string_lossy(),
                };

                Err(format!("cuLaunchKernel failed: {message} (error {e})"))
            }
        }
    }
}
