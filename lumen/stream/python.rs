use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate as core;
use crate::tensor::python::PyTensor;

#[pyclass(name = "Stream", module = "lumen", frozen)]
pub(crate) struct PyStream {
    device: core::Device,
    stream_id: usize,
    inner: std::sync::Arc<dyn core::stream::Stream>,
}

/// A Metal kernel of your own, compiled by an MPS stream (`Stream.compile`),
/// for its `launch`.
#[pyclass(name = "Kernel", module = "lumen", frozen)]
pub(crate) struct PyKernel {
    key: String,
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl PyKernel {
    fn __repr__(&self) -> String {
        format!("<{}.Kernel {}>", crate::LIBRARY_NAME, self.name)
    }
}

static COMPILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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

    #[pyo3(signature = (source, name = None))]
    fn compile(&self, source: &str, name: Option<&str>) -> PyResult<PyKernel> {
        self.mps("compile")?;
        let n = COMPILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = format!("{}.Stream.compile#{n}", crate::LIBRARY_NAME);
        let (key, name) = compile_kernel(source, name, &key)?;
        Ok(PyKernel { key, name })
    }

    #[pyo3(signature = (kernel, tensors, args = Vec::new(), *, grid = None, threadgroups = None))]
    fn launch(
        &self,
        kernel: PyRef<'_, PyKernel>,
        tensors: Vec<PyRef<'_, PyTensor>>,
        args: Vec<Bound<'_, PyAny>>,
        grid: Option<Vec<usize>>,
        threadgroups: Option<Vec<usize>>,
    ) -> PyResult<()> {
        self.mps("launch")?;
        let groups = threadgroups.is_some();
        let Some(mut sizes) = (match (grid, threadgroups) {
            (Some(sizes), None) | (None, Some(sizes)) => Some(sizes),
            _ => None,
        }) else {
            return Err(PyTypeError::new_err(
                "launch takes one of grid and threadgroups",
            ));
        };
        sizes.resize(3, 1);
        let args = args
            .iter()
            .map(|a| match a.cast::<PyBytes>() {
                Ok(bytes) => Ok(bytes.as_bytes().to_vec()),
                Err(_) => a.call_method0("tobytes")?.extract::<Vec<u8>>(),
            })
            .collect::<PyResult<Vec<Vec<u8>>>>()?;
        let name = &kernel.name;
        if let Some(t) = tensors.iter().find(|t| t.inner.device() != self.device) {
            return Err(PyValueError::new_err(format!(
                "{name}: its tensors must be on {}, got one on {}",
                self.device,
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
            let sizes = [sizes[0], sizes[1], sizes[2]];
            let grid = if groups {
                Grid::Groups(sizes)
            } else {
                Grid::Threads(sizes)
            };
            let label = core::graph::intern(name.clone());
            launch(&kernel.key, &buffers, &args, grid, keep, label)
                .map_err(PyRuntimeError::new_err)?;
        }
        #[cfg(not(lumen_mps_linked))]
        let _ = (args, groups, sizes);
        Ok(())
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
    fn lumen_mps_compile_kernel(
        source: *const std::ffi::c_char,
        name: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
        chosen: *mut std::ffi::c_char,
        len: usize,
    ) -> i32;
    fn lumen_mps_buffer_pointer(ptr: *const u8, offset: *mut usize) -> usize;
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
    m.add_class::<PyStream>()?;
    m.add_class::<PyKernel>()
}
