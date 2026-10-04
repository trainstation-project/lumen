//! `lumen.mps.synchronize` and `lumen.cuda.synchronize`: bindings for the
//! device streams, registered into `lumen._C` by [`crate::python`].

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;
use crate::tensor::python::PyTensor;

/// Wait for all work on the MPS stream (`torch.mps.synchronize()`).
#[pyfunction]
fn _mps_synchronize(py: Python<'_>) -> PyResult<()> {
    core::allocator::allocator_for(core::Device::Mps).map_err(PyRuntimeError::new_err)?;
    #[cfg(lumen_mps_linked)]
    py.detach(core::stream::mps::synchronize);
    #[cfg(not(lumen_mps_linked))]
    let _ = py;
    Ok(())
}

/// Wait for all work on a CUDA device (`torch.cuda.synchronize(device)`):
/// `device` is an index, a CUDA device, or `None` for device 0.
#[pyfunction]
#[pyo3(signature = (device = None))]
fn _cuda_synchronize(py: Python<'_>, device: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let device = match device {
        None => core::Device::Cuda(0),
        Some(d) => match d.extract::<usize>() {
            Ok(index) => core::Device::Cuda(index),
            Err(_) => crate::python::resolve_device(Some(d))?,
        },
    };
    let core::Device::Cuda(index) = device else {
        return Err(PyValueError::new_err(format!(
            "Expected a cuda device, but got: {device}"
        )));
    };
    core::allocator::allocator_for(device).map_err(PyRuntimeError::new_err)?;
    #[cfg(lumen_cuda_linked)]
    py.detach(|| core::stream::cuda::synchronize(index));
    #[cfg(not(lumen_cuda_linked))]
    let _ = (py, index);
    Ok(())
}

/// Whether a CUDA device can hold tensors in this build
/// (`torch.cuda.is_available()`).
#[pyfunction]
fn _cuda_is_available() -> bool {
    core::device::cuda::is_available()
}

/// Make the CUDA stream `stream` (a `cudaStream_t` as an int) wait for the
/// work queued so far on lumen's stream on device `index`, without blocking
/// the host. Backs `Tensor.__dlpack__(stream=...)`.
#[pyfunction]
fn _cuda_stream_wait(index: usize, stream: usize) -> PyResult<()> {
    core::allocator::allocator_for(core::Device::Cuda(index)).map_err(PyRuntimeError::new_err)?;
    #[cfg(lumen_cuda_linked)]
    core::stream::cuda::wait(index, stream);
    #[cfg(not(lumen_cuda_linked))]
    let _ = stream;
    Ok(())
}

#[cfg(lumen_mps_linked)]
unsafe extern "C" {
    // In lumen/ops/mps/shim.mm and lumen/allocator/mps/mps_shim.mm.
    fn lumen_mps_compile_kernel(
        source: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        chosen: *mut std::ffi::c_char,
        len: usize,
    ) -> i32;
    fn lumen_mps_buffer_pointer(ptr: *const u8, offset: *mut usize) -> usize;
}

/// The MPS stream, or why there is none.
fn mps() -> PyResult<()> {
    core::allocator::allocator_for(core::Device::Mps).map_err(PyRuntimeError::new_err)?;
    if cfg!(lumen_mps_linked) {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err("lumen was built without MPS"))
    }
}

/// Compiled kernels so far: each one's key is new.
static COMPILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Compile Metal `source`'s kernel `name` (or its only one): its key, for
/// `_mps_launch` (its own, so kernels of one name never collide), and name.
#[pyfunction]
#[pyo3(signature = (source, name = None))]
fn _mps_compile(source: &str, name: Option<&str>) -> PyResult<(String, String)> {
    mps()?;
    let n = COMPILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    compile_kernel(source, name, &format!("lumen.mps.compile#{n}"))
}

#[cfg(lumen_mps_linked)]
fn compile_kernel(source: &str, name: Option<&str>, key: &str) -> PyResult<(String, String)> {
    use std::ffi::{CStr, CString};
    let c = |s: &str| CString::new(s).map_err(|e| PyValueError::new_err(e.to_string()));
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
        0 => Ok((key.to_owned(), chosen)),
        -1 => Err(PyRuntimeError::new_err(
            "compiling the Metal source failed (Metal's error is in the log)",
        )),
        -2 => Err(PyRuntimeError::new_err(format!(
            "making the Metal pipeline of {chosen} failed (Metal's error is in the log)"
        ))),
        -3 => Err(PyValueError::new_err(format!(
            "the Metal source has several kernels ({chosen}): name the one to compile"
        ))),
        _ => Err(PyValueError::new_err(format!(
            "the Metal source has no kernel named {:?}: it has {chosen}",
            name.unwrap_or_default()
        ))),
    }
}

#[cfg(not(lumen_mps_linked))]
fn compile_kernel(_source: &str, _name: Option<&str>, _key: &str) -> PyResult<(String, String)> {
    Err(PyRuntimeError::new_err("lumen was built without MPS"))
}

/// Encode kernel `key` (from `_mps_compile`; profiled as `name`) into the MPS
/// stream over `grid` (threads, or threadgroups of 16x16 if `groups`), its
/// buffers `tensors`' memory in order (each from its first element), then
/// `args`' bytes.
#[pyfunction]
fn _mps_launch(
    key: &str,
    name: &str,
    tensors: Vec<PyRef<'_, PyTensor>>,
    args: Vec<Vec<u8>>,
    grid: (usize, usize, usize),
    groups: bool,
) -> PyResult<()> {
    mps()?;
    if let Some(t) = tensors
        .iter()
        .find(|t| t.inner.device() != core::Device::Mps)
    {
        return Err(PyValueError::new_err(format!(
            "{name}: its tensors must be on MPS, got one on {}",
            t.inner.device()
        )));
    }
    #[cfg(lumen_mps_linked)]
    {
        use core::ops::mps::{Grid, launch};
        let buffers: Vec<*const u8> = tensors
            .iter()
            .map(|t| t.inner.data_ptr().cast_const())
            .collect();
        let keep = tensors.iter().map(|t| t.inner.clone()).collect();
        let grid = [grid.0, grid.1, grid.2];
        let grid = if groups {
            Grid::Groups(grid)
        } else {
            Grid::Threads(grid)
        };
        let label = core::graph::intern(name.to_owned());
        launch(key, &buffers, &args, grid, keep, label).map_err(PyRuntimeError::new_err)?;
    }
    #[cfg(not(lumen_mps_linked))]
    let _ = (key, tensors, args, grid, groups);
    Ok(())
}

/// The MPS stream's open `MTLCommandBuffer`, as a pointer (see
/// `lumen.mps.command_buffer`).
#[pyfunction]
fn _mps_command_buffer() -> PyResult<usize> {
    mps()?;
    Ok(command_buffer())
}

#[cfg(lumen_mps_linked)]
fn command_buffer() -> usize {
    core::stream::mps::command_buffer() as usize
}

#[cfg(not(lumen_mps_linked))]
fn command_buffer() -> usize {
    0
}

/// The `MTLBuffer` holding MPS tensor `t`'s first element, as a pointer, and
/// that element's byte offset in it.
#[pyfunction]
fn _mps_buffer(t: PyRef<'_, PyTensor>) -> PyResult<(usize, usize)> {
    mps()?;
    if t.inner.device() != core::Device::Mps {
        return Err(PyValueError::new_err(format!(
            "an MPS tensor has an MTLBuffer, got one on {}",
            t.inner.device()
        )));
    }
    buffer_of(&t.inner)
}

#[cfg(lumen_mps_linked)]
fn buffer_of(t: &core::Tensor) -> PyResult<(usize, usize)> {
    let mut offset = 0;
    let buffer = unsafe { lumen_mps_buffer_pointer(t.data_ptr().cast_const(), &mut offset) };
    if buffer == 0 {
        return Err(PyRuntimeError::new_err("the tensor is in no MTLBuffer"));
    }
    Ok((buffer, offset))
}

#[cfg(not(lumen_mps_linked))]
fn buffer_of(_t: &core::Tensor) -> PyResult<(usize, usize)> {
    Ok((0, 0))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_mps_compile, m)?)?;
    m.add_function(wrap_pyfunction!(_mps_launch, m)?)?;
    m.add_function(wrap_pyfunction!(_mps_command_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(_mps_buffer, m)?)?;
    m.add_function(wrap_pyfunction!(_mps_synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(_cuda_synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(_cuda_is_available, m)?)?;
    m.add_function(wrap_pyfunction!(_cuda_stream_wait, m)?)
}
