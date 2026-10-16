//! Metal kernels of your own, for a custom op (`lumen.custom_op`) to
//! launch into lumen's MPS stream, in order with a compiled function's
//! steps.

use crate::{Device, Tensor};

#[cfg(feature = "python")]
pub(crate) mod python;
#[cfg(test)]
mod tests;

/// How a kernel's threads are laid out.
#[derive(Debug, Clone, Copy)]
pub enum Grid {
    /// A thread per index of an x-by-y-by-z grid (x varying fastest).
    Threads([usize; 3]),
    /// That many threadgroups of 16x16 threads.
    Groups([usize; 3]),
}

pub struct Kernel {
    key: String,
    name: String,
}

static COMPILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Kernel {
    /// Compile `source`'s kernel `name` (its only one if `None`).
    pub fn new(source: &str, name: Option<&str>) -> Result<Self, String> {
        let n = COMPILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = format!("{}.custom_op.Kernel#{n}", crate::LIBRARY_NAME);
        let name = compile(source, name, &key)?;
        Ok(Kernel { key, name })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Launch it over `grid` into lumen's MPS stream (no wait), on
    /// `tensors`' memory (from their first elements) then `args`' bytes.
    pub fn launch(&self, tensors: &[Tensor], args: &[Vec<u8>], grid: Grid) -> Result<(), String> {
        if let Some(t) = tensors
            .iter()
            .find(|t| !matches!(t.device(), Device::Mps(_)))
        {
            return Err(format!(
                "{}: its tensors must be on mps, got one on {}",
                self.name,
                t.device()
            ));
        }
        #[cfg(lumen_mps_linked)]
        {
            let buffers: Vec<*const u8> =
                tensors.iter().map(|t| t.data_ptr().cast_const()).collect();
            let label = crate::graph::intern(self.name.clone());
            crate::ops::mps::launch(&self.key, &buffers, args, grid, tensors.to_vec(), label)
        }
        #[cfg(not(lumen_mps_linked))]
        {
            let _ = (args, grid);
            Ok(())
        }
    }
}

#[cfg(lumen_mps_linked)]
unsafe extern "C" {
    // In shim.mm.
    fn lumen_mps_compile_kernel(
        source: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        chosen: *mut std::ffi::c_char,
        len: usize,
    ) -> i32;
}

#[cfg(lumen_mps_linked)]
fn compile(source: &str, name: Option<&str>, key: &str) -> Result<String, String> {
    use std::ffi::{CStr, CString};
    let c = |s: &str| CString::new(s).map_err(|e| e.to_string());
    let (source, wanted, c_key) = (c(source)?, name.map(c).transpose()?, c(key)?);
    let mut chosen = vec![0 as std::ffi::c_char; 4096];
    let status = unsafe {
        lumen_mps_compile_kernel(
            source.as_ptr(),
            wanted.as_ref().map_or(std::ptr::null(), |w| w.as_ptr()),
            c_key.as_ptr(),
            chosen.as_mut_ptr(),
            chosen.len(),
        )
    };
    let chosen = unsafe { CStr::from_ptr(chosen.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    match status {
        0 => Ok(chosen),
        -1 => Err("compiling the Metal source failed (Metal's error is in the log)".into()),
        -2 => Err(format!(
            "making the Metal pipeline of {chosen} failed (Metal's error is in the log)"
        )),
        -3 => Err(format!(
            "the Metal source has several kernels ({chosen}): name the one to compile"
        )),
        _ => Err(format!(
            "the Metal source has no kernel named {:?}: it has {chosen}",
            name.unwrap_or_default()
        )),
    }
}

#[cfg(not(lumen_mps_linked))]
fn compile(_source: &str, _name: Option<&str>, _key: &str) -> Result<String, String> {
    Err("lumen was built without MPS".into())
}
