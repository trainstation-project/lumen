//! Python bindings for `lumen.Stream` and `lumen.Event`.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate as core;
use crate::tensor::python::PyTensor;

/// Python `lumen.Stream`: lumen's stream on a device, or an external CUDA stream.
#[pyclass(name = "Stream", module = "lumen", frozen)]
pub(crate) struct PyStream {
    device: core::Device,
    /// 0 for lumen's stream; otherwise an external `cudaStream_t` as an integer.
    stream_id: usize,
    inner: std::sync::Arc<dyn core::stream::Stream>,
}

impl PyStream {
    /// Blocks the host until this stream's work finishes; an external CUDA stream waits on
    /// that stream alone rather than the whole device.
    fn wait_host(&self) {
        #[cfg(lumen_cuda_linked)]
        if let (core::Device::Cuda(index), id @ 1..) = (self.device, self.stream_id) {
            return core::stream::cuda::synchronize_external(index, id);
        }

        self.inner.synchronize();
    }

    /// Errors unless this is an MPS stream in a build with MPS; `what` names the caller.
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
    /// Opens the stream on `device` (the default device when `None`); a nonzero `stream_id`
    /// wraps an external stream and is allowed only on CUDA.
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

    /// The device this stream runs on.
    #[getter]
    fn device(&self) -> crate::device::python::PyDevice {
        crate::device::python::PyDevice::from_core(self.device)
    }

    /// 0 for lumen's stream, otherwise the external stream handle.
    #[getter]
    fn stream_id(&self) -> usize {
        self.stream_id
    }

    /// Blocks until this stream's work finishes, releasing the GIL while waiting.
    fn synchronize(&self, py: Python<'_>) {
        let _op = core::profiler::record_op(op_name!("synchronize"), Vec::new);
        py.detach(|| self.wait_host());
    }

    /// Makes later work on this stream wait for `other`'s work so far: an external CUDA stream
    /// waits on lumen's stream on the same device without blocking the host; otherwise the host
    /// blocks until `other` finishes.
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

    /// Streams are equal when they share a device and stream id.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyStream>()
            .is_ok_and(|o| (o.get().device, o.get().stream_id) == (self.device, self.stream_id))
    }

    /// Hashes the device and stream id, matching `__eq__`.
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.device, self.stream_id).hash(&mut hasher);
        hasher.finish()
    }

    /// The `MTLCommandBuffer` lumen is encoding into, as an integer pointer (MPS only).
    fn command_buffer(&self) -> PyResult<usize> {
        self.mps("command_buffer")?;
        Ok(command_buffer())
    }

    /// The `MTLBuffer` holding `tensor`'s data and the byte offset of its data in it (MPS only).
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

    /// Shows the device, and the stream id in hex for an external stream.
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
    /// Returns the `MTLBuffer` containing `ptr` and writes `ptr`'s offset in it, or 0 when none does.
    fn lumen_mps_buffer_pointer(ptr: *const u8, offset: *mut usize) -> usize;
}

/// The current `MTLCommandBuffer` as an integer pointer.
#[cfg(lumen_mps_linked)]
fn command_buffer() -> usize {
    core::stream::mps::command_buffer() as usize
}

/// Without MPS there is no command buffer; never reached since `mps` errors first.
#[cfg(not(lumen_mps_linked))]
fn command_buffer() -> usize {
    0
}

/// The `MTLBuffer` pointer and byte offset holding `t`'s data; errors when no buffer holds it.
#[cfg(lumen_mps_linked)]
fn buffer_of(t: &core::Tensor) -> PyResult<(usize, usize)> {
    let mut offset = 0;
    let buffer = unsafe { lumen_mps_buffer_pointer(t.data_ptr().cast_const(), &mut offset) };

    if buffer == 0 {
        return Err(PyRuntimeError::new_err("the tensor is in no MTLBuffer"));
    }

    Ok((buffer, offset))
}

/// Without MPS there is no buffer; never reached since `mps` errors first.
#[cfg(not(lumen_mps_linked))]
fn buffer_of(_t: &core::Tensor) -> PyResult<(usize, usize)> {
    Ok((0, 0))
}

/// Python `lumen.Event`: a marker recorded into a device's stream, optionally timed.
#[pyclass(name = "Event", module = "lumen")]
pub(crate) struct PyEvent {
    inner: Box<dyn core::stream::Event>,
}

#[pymethods]
impl PyEvent {
    /// Creates an unrecorded event on `device` (the default device when `None`).
    #[new]
    #[pyo3(signature = (device = None, *, enable_timing = false))]
    fn new(device: Option<&Bound<'_, PyAny>>, enable_timing: bool) -> PyResult<Self> {
        let device = crate::device::python::resolve_device(device)?;

        let inner =
            core::stream::event_for(device, enable_timing).map_err(PyRuntimeError::new_err)?;

        Ok(PyEvent { inner })
    }

    /// The device whose stream the event records into.
    #[getter]
    fn device(&self) -> crate::device::python::PyDevice {
        crate::device::python::PyDevice::from_core(self.inner.device())
    }

    /// Whether the event records a timestamp for `elapsed_time`.
    #[getter]
    fn enable_timing(&self) -> bool {
        self.inner.timing()
    }

    /// Records the event at the current end of the device's stream.
    fn record(&mut self) -> PyResult<()> {
        self.inner.record().map_err(PyRuntimeError::new_err)
    }

    /// Whether the work before the recorded point has finished, without blocking.
    fn query(&self) -> bool {
        self.inner.query()
    }

    /// Blocks until the work before the recorded point finishes, releasing the GIL.
    fn synchronize(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;

        py.detach(|| inner.synchronize())
            .map_err(PyRuntimeError::new_err)
    }

    /// Milliseconds between this event and `end`; both need timing enabled and must have completed.
    fn elapsed_time(&self, py: Python<'_>, end: PyRef<'_, PyEvent>) -> PyResult<f64> {
        let (start, end) = (&self.inner, &end.inner);

        py.detach(|| start.elapsed_time(end.as_ref()))
            .map_err(PyRuntimeError::new_err)
    }

    /// Shows the device and whether timing is enabled.
    fn __repr__(&self) -> String {
        format!(
            "{}.Event(device='{}', enable_timing={})",
            crate::LIBRARY_NAME,
            self.inner.device(),
            if self.inner.timing() { "True" } else { "False" }
        )
    }
}

/// Adds `Stream` and `Event` to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStream>()?;
    m.add_class::<PyEvent>()
}
