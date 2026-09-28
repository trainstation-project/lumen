//! `lumen.mps.synchronize` and `lumen.cuda.synchronize`: bindings for the
//! device streams, registered into `lumen._C` by [`crate::python`].

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;

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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_mps_synchronize, m)?)?;
    m.add_function(wrap_pyfunction!(_cuda_synchronize, m)?)
}
