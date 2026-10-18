//! Google TPUs, through libtpu's PJRT C API (`pjrt.c`, which loads libtpu at
//! run time: `TPU_LIBRARY_PATH`, else the loader's search). One PJRT client
//! for the process, created on first use; its addressable devices are
//! `tpu:0`, `tpu:1`, ... It compiles and runs the programs of
//! `lumen/ops/tpu`. Built with the `tpu` feature on Linux
//! (`cfg(lumen_tpu_linked)`); without it, or without libtpu or TPUs, there
//! are none.

use super::DeviceBackend;

pub mod layout;

#[cfg(lumen_tpu_linked)]
pub(crate) mod ffi {
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub fn lumen_tpu_device_count() -> i32;
        pub fn lumen_tpu_error() -> *const c_char;
        pub fn lumen_tpu_alloc(index: i32, nbytes: usize, address: *mut usize) -> *mut c_void;
        pub fn lumen_tpu_free(buffer: *mut c_void);
        pub fn lumen_tpu_read(
            buffer: *mut c_void,
            size: usize,
            offset: usize,
            dst: *mut c_void,
            nbytes: usize,
        ) -> i32;
        pub fn lumen_tpu_compile(code: *const c_char, size: usize) -> *mut c_void;
        pub fn lumen_tpu_run(
            executable: *mut c_void,
            index: i32,
            scalars: *const i32,
            count: usize,
            host: *const c_void,
            nbytes: usize,
            region: *mut *mut c_void,
        ) -> i32;
        pub fn lumen_tpu_wait(buffer: *mut c_void) -> i32;
        pub fn lumen_tpu_execute_ns() -> u64;
        #[cfg(test)]
        pub fn lumen_tpu_memory(index: i32, in_use: *mut i64, limit: *mut i64) -> i32;
    }
}

/// Why the last TPU call failed (or no TPU was found).
#[cfg(lumen_tpu_linked)]
pub(crate) fn error() -> String {
    // SAFETY: the shim's NUL-terminated message buffer.
    unsafe { std::ffi::CStr::from_ptr(ffi::lumen_tpu_error()) }
        .to_string_lossy()
        .into_owned()
}

/// Device `index`'s memory in use and its limit, in bytes.
#[cfg(all(test, lumen_tpu_linked))]
pub(crate) fn memory(index: usize) -> Result<(usize, usize), String> {
    let (mut in_use, mut limit) = (0, 0);
    // SAFETY: two writable counts.
    match unsafe { ffi::lumen_tpu_memory(index as i32, &mut in_use, &mut limit) } {
        0 => Ok((in_use.max(0) as usize, limit.max(0) as usize)),
        _ => Err(error()),
    }
}

/// Held by each test of a TPU: some measure its memory, which the others
/// would change.
#[cfg(all(test, lumen_tpu_linked))]
pub(crate) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct Tpu;
pub static TPU: &dyn DeviceBackend = &Tpu;

impl DeviceBackend for Tpu {
    fn name(&self) -> &'static str {
        "tpu"
    }

    fn device_count(&self) -> usize {
        #[cfg(lumen_tpu_linked)]
        {
            // SAFETY: no arguments; loads libtpu once.
            unsafe { ffi::lumen_tpu_device_count() }.max(0) as usize
        }
        #[cfg(not(lumen_tpu_linked))]
        {
            0
        }
    }

    fn check(&self, index: usize) -> Result<(), String> {
        let count = self.device_count();
        if index < count {
            return Ok(());
        }
        #[cfg(lumen_tpu_linked)]
        let why = match count {
            0 => format!("; {}", error()),
            _ => String::new(),
        };
        #[cfg(not(lumen_tpu_linked))]
        let why = "; lumen was built without TPU support (the `tpu` feature, on Linux)";
        Err(format!(
            "TPU device {index} is not available ({count} device(s) found{why})"
        ))
    }
}
