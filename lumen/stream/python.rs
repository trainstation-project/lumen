use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;
use crate::tensor::python::PyTensor;

#[pyclass(name = "Stream", module = "lumen", frozen)]
pub(crate) struct PyStream {
    device: core::Device,
    inner: std::sync::Arc<dyn core::stream::Stream>,
}

#[pymethods]
impl PyStream {
    #[new]
    #[pyo3(signature = (device = None))]
    fn new(device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let device = crate::device::python::resolve_device(device)?;
        let inner = core::stream::stream_for(device).map_err(PyRuntimeError::new_err)?;
        Ok(PyStream { device, inner })
    }

    #[getter]
    fn device(&self) -> String {
        self.device.to_string()
    }

    fn synchronize(&self, py: Python<'_>) {
        let _op = core::profiler::record_op(op_name!("synchronize"), Vec::new);
        py.detach(|| self.inner.synchronize());
    }

    fn __repr__(&self) -> String {
        format!("{}.Stream(device='{}')", crate::LIBRARY_NAME, self.device)
    }
}

#[pyfunction]
fn _cuda_stream_wait(index: usize, stream: usize) -> PyResult<()> {
    core::stream::stream_for(core::Device::Cuda(index)).map_err(PyRuntimeError::new_err)?;
    #[cfg(lumen_cuda_linked)]
    core::stream::CudaStream { index }.wait(stream);
    #[cfg(not(lumen_cuda_linked))]
    let _ = stream;
    Ok(())
}

#[cfg(lumen_mps_linked)]
unsafe extern "C" {
    fn lumen_mps_compile_kernel(
        source: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        chosen: *mut std::ffi::c_char,
        len: usize,
    ) -> i32;
    fn lumen_mps_buffer_pointer(ptr: *const u8, offset: *mut usize) -> usize;
}

fn mps() -> PyResult<()> {
    core::allocator::allocator_for(core::Device::Mps(0)).map_err(PyRuntimeError::new_err)?;
    if cfg!(lumen_mps_linked) {
        Ok(())
    } else {
        Err(PyRuntimeError::new_err("lumen was built without MPS"))
    }
}

static COMPILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
        .find(|t| t.inner.device() != core::Device::Mps(0))
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

#[pyfunction]
fn _mps_buffer(t: PyRef<'_, PyTensor>) -> PyResult<(usize, usize)> {
    mps()?;
    if t.inner.device() != core::Device::Mps(0) {
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
    m.add_class::<PyStream>()?;
    m.add_function(wrap_pyfunction!(_cuda_stream_wait, m)?)
}
