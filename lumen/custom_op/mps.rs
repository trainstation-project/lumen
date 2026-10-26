//! User-written Metal kernels compiled at runtime and launched on lumen's MPS stream.

use crate::{Device, Tensor};

/// Dispatch size of a Metal kernel launch.
#[derive(Debug, Clone, Copy)]
pub enum Grid {
    /// Total threads per dimension; Metal picks the threadgroup size.
    Threads([usize; 3]),
    /// Threadgroups per dimension, each of the pipeline's maximum size.
    Groups([usize; 3]),
    /// `groups` threadgroups of `threads` threads each.
    GroupsOf { groups: [usize; 3], threads: usize },
    /// A 1-D grid of `threads` threads in threadgroups of `group` threads.
    Elementwise { threads: usize, group: usize },
}

/// A compiled Metal compute pipeline from user source.
pub struct MetalKernel {
    #[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
    /// Name the pipeline is registered under in the MPS kernel cache.
    key: String,
    /// Name of the kernel function chosen from the source.
    name: String,
}

/// Counter giving each compiled kernel a unique cache key.
static COMPILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl MetalKernel {
    /// Compiles `source`; `name` picks the kernel when the source defines several.
    pub fn new(source: &str, name: Option<&str>) -> Result<Self, String> {
        let n = COMPILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = format!("{}.custom_op.MetalKernel#{n}", crate::LIBRARY_NAME);
        let name = compile(source, name, &key)?;
        Ok(MetalKernel { key, name })
    }

    /// Name of the compiled kernel function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Enqueues the kernel with `tensors` bound as buffers then `args` as constant bytes.
    /// Errors when a tensor isn't on MPS; a no-op when lumen is built without MPS.
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
            crate::kernels::mps::launch(&self.key, &buffers, args, grid, tensors.to_vec(), label)
        }
        #[cfg(not(lumen_mps_linked))]
        {
            let _ = (args, grid);
            Ok(())
        }
    }
}

// Implemented in the Objective-C MPS runtime.
#[cfg(lumen_mps_linked)]
unsafe extern "C" {
    /// Compiles `source` and registers its pipeline under `key`, writing the chosen kernel's name (or
    /// the available names on error) into `chosen`. Returns 0 on success or a negative error code.
    fn lumen_mps_compile_kernel(
        source: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        chosen: *mut std::ffi::c_char,
        len: usize,
    ) -> i32;
}

#[cfg(lumen_mps_linked)]
/// Compiles `source` into a pipeline cached under `key`; returns the chosen kernel's name.
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

/// Fails: lumen was built without MPS.
#[cfg(not(lumen_mps_linked))]
fn compile(_source: &str, _name: Option<&str>, _key: &str) -> Result<String, String> {
    Err("lumen was built without MPS".into())
}
