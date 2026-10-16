use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;
use crate::tensor::python::PyTensor;

#[pyclass(name = "Stream", module = "lumen", frozen)]
pub(crate) struct PyStream {
    device: core::Device,
    stream_id: usize,
    inner: std::sync::Arc<dyn core::stream::Stream>,
}

impl PyStream {
    fn wait_host(&self) {
        #[cfg(lumen_cuda_linked)]
        if let (core::Device::Cuda(index), id @ 1..) = (self.device, self.stream_id) {
            return core::stream::cuda::synchronize_external(index, id);
        }
        self.inner.synchronize();
    }

    fn mps(&self, what: &str) -> PyResult<()> {
        match (self.device, cfg!(lumen_mps_linked)) {
            (core::Device::Mps(_), true) => Ok(()),
            (core::Device::Mps(_), false) => {
                Err(PyRuntimeError::new_err("lumen was built without MPS"))
            }
            (device, _) => Err(PyValueError::new_err(format!(
                "{what}: {device}'s stream runs no Metal kernels (an MPS stream does)"
            ))),
        }
    }
}

#[pymethods]
impl PyStream {
    #[new]
    #[pyo3(signature = (device = None, *, stream_id = 0))]
    fn new(device: Option<&Bound<'_, PyAny>>, stream_id: usize) -> PyResult<Self> {
        let device = crate::device::python::resolve_device(device)?;
        let inner = core::stream::stream_for(device).map_err(PyRuntimeError::new_err)?;
        if stream_id != 0 && !matches!(device, core::Device::Cuda(_)) {
            return Err(PyValueError::new_err(format!(
                "stream_id: {device} has only lumen's stream (an external stream is CUDA's)"
            )));
        }
        Ok(PyStream {
            device,
            stream_id,
            inner,
        })
    }

    #[getter]
    fn device(&self) -> crate::device::python::PyDevice {
        crate::device::python::PyDevice::from_core(self.device)
    }

    #[getter]
    fn stream_id(&self) -> usize {
        self.stream_id
    }

    fn synchronize(&self, py: Python<'_>) {
        let _op = core::profiler::record_op(op_name!("synchronize"), Vec::new);
        py.detach(|| self.wait_host());
    }

    fn wait_stream(&self, py: Python<'_>, other: PyRef<'_, PyStream>) {
        if self.device == other.device && self.stream_id == other.stream_id {
            return;
        }
        #[cfg(lumen_cuda_linked)]
        if let (core::Device::Cuda(index), 0) = (other.device, other.stream_id)
            && self.device == other.device
        {
            return core::stream::CudaStream { index }.wait(self.stream_id);
        }
        let other: &PyStream = &other;
        py.detach(|| other.wait_host());
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyStream>()
            .is_ok_and(|o| (o.get().device, o.get().stream_id) == (self.device, self.stream_id))
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.device, self.stream_id).hash(&mut hasher);
        hasher.finish()
    }

    fn command_buffer(&self) -> PyResult<usize> {
        self.mps("command_buffer")?;
        Ok(command_buffer())
    }

    fn buffer(&self, tensor: PyRef<'_, PyTensor>) -> PyResult<(usize, usize)> {
        self.mps("buffer")?;
        if tensor.inner.device() != self.device {
            return Err(PyValueError::new_err(format!(
                "an MPS tensor has an MTLBuffer, got one on {}",
                tensor.inner.device()
            )));
        }
        buffer_of(&tensor.inner)
    }

    fn __repr__(&self) -> String {
        match self.stream_id {
            0 => format!("{}.Stream(device='{}')", crate::LIBRARY_NAME, self.device),
            id => format!(
                "{}.Stream(device='{}', stream_id={id:#x})",
                crate::LIBRARY_NAME,
                self.device
            ),
        }
    }
}

#[cfg(lumen_mps_linked)]
unsafe extern "C" {
    fn lumen_mps_buffer_pointer(ptr: *const u8, offset: *mut usize) -> usize;
}

#[cfg(lumen_mps_linked)]
fn command_buffer() -> usize {
    core::stream::mps::command_buffer() as usize
}

#[cfg(not(lumen_mps_linked))]
fn command_buffer() -> usize {
    0
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
    m.add_class::<PyStream>()
}
