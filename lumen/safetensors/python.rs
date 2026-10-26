//! Python bindings for `lumen.safetensors`: serialize, save, deserialize and `safe_open`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyKeyError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::{Error, SafeTensorsFile};
use crate::device::python::resolve_device;
use crate::tensor::python::PyTensor;
use crate::{Device, Tensor};

/// Maps I/O errors to `OSError`, missing tensors to `KeyError` and the rest to `ValueError`.
fn to_py(e: Error) -> PyErr {
    match e {
        Error::Io(e) => PyOSError::new_err(e.to_string()),
        Error::TensorNotFound(name) => PyKeyError::new_err(name),
        e => PyValueError::new_err(e.to_string()),
    }
}

/// `(name, tensor)` pairs as passed from Python.
type Named<'py> = Vec<(String, PyRef<'py, PyTensor>)>;

/// Borrows Python `(name, tensor)` pairs as core ones.
fn named<'a>(tensors: &'a Named<'_>) -> Vec<(&'a str, &'a Tensor)> {
    tensors
        .iter()
        .map(|(n, t)| (n.as_str(), &t.inner))
        .collect()
}

/// Wraps loaded tensors for Python.
fn wrap(tensors: Vec<(String, Tensor)>) -> Vec<(String, PyTensor)> {
    tensors
        .into_iter()
        .map(|(n, t)| (n, PyTensor::wrap(t)))
        .collect()
}

/// Serializes named tensors and optional string metadata into safetensors `bytes`.
#[pyfunction]
#[pyo3(signature = (tensors, metadata = None))]
fn _safetensors_serialize<'py>(
    py: Python<'py>,
    tensors: Named<'_>,
    metadata: Option<BTreeMap<String, String>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = super::serialize(&named(&tensors), metadata).map_err(to_py)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Writes named tensors and optional string metadata to a safetensors file at `path`.
#[pyfunction]
#[pyo3(signature = (tensors, path, metadata = None))]
fn _safetensors_save_file(
    tensors: Named<'_>,
    path: PathBuf,
    metadata: Option<BTreeMap<String, String>>,
) -> PyResult<()> {
    super::save_file(&named(&tensors), metadata, path).map_err(to_py)
}

/// Loads every tensor from safetensors `bytes` onto `device` (the default device when `None`).
#[pyfunction]
#[pyo3(signature = (data, device = None))]
fn _safetensors_deserialize(
    data: &[u8],
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, PyTensor)>> {
    let device = resolve_device(device)?;
    super::deserialize(data, device).map(wrap).map_err(to_py)
}

/// Python `safe_open`: a context manager that reads tensors from a safetensors file by name.
#[pyclass(name = "safe_open", module = "lumen.safetensors")]
struct PySafeOpen {
    /// `None` once closed.
    file: Option<SafeTensorsFile>,
    /// Where tensors load unless `get_tensor` names another device.
    device: Device,
}

impl PySafeOpen {
    /// The open file, or a `ValueError` once closed.
    fn file(&self) -> PyResult<&SafeTensorsFile> {
        self.file
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("the file is closed"))
    }
}

#[pymethods]
impl PySafeOpen {
    /// Opens `filename` and reads its header; tensors load onto `device` (the default when `None`).
    #[new]
    #[pyo3(signature = (filename, device = None))]
    fn new(filename: PathBuf, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let device = resolve_device(device)?;
        let file = SafeTensorsFile::open(filename).map_err(to_py)?;
        Ok(Self {
            file: Some(file),
            device,
        })
    }

    /// Tensor names in sorted order.
    fn keys(&self) -> PyResult<Vec<String>> {
        let mut names = self.offset_keys()?;
        names.sort();
        Ok(names)
    }

    /// Tensor names in the order their data appears in the file.
    fn offset_keys(&self) -> PyResult<Vec<String>> {
        let tensors = self.file()?.metadata().tensors();
        Ok(tensors.iter().map(|(n, _)| n.clone()).collect())
    }

    /// The header's `__metadata__` string map, or `None` when it has none.
    fn metadata(&self) -> PyResult<Option<BTreeMap<String, String>>> {
        Ok(self.file()?.metadata().metadata().cloned())
    }

    /// Loads tensor `name` onto `device`, or the device given at open when `None`.
    #[pyo3(signature = (name, device = None))]
    fn get_tensor(&self, name: &str, device: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
        let device = match device {
            Some(d) if !d.is_none() => resolve_device(Some(d))?,
            _ => self.device,
        };
        let t = self.file()?.tensor(name, device).map_err(to_py)?;
        Ok(PyTensor::wrap(t))
    }

    /// Loads every tensor, in data order, onto the device given at open.
    fn get_tensors(&self) -> PyResult<Vec<(String, PyTensor)>> {
        self.file()?.tensors(self.device).map(wrap).map_err(to_py)
    }

    /// Closes the file; later reads raise `ValueError`.
    fn close(&mut self) {
        self.file = None;
    }

    /// Returns the object itself for `with`.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Closes the file on leaving `with`.
    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close();
    }
}

/// Adds the safetensors functions and `safe_open` to the Python module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_safetensors_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(_safetensors_save_file, m)?)?;
    m.add_function(wrap_pyfunction!(_safetensors_deserialize, m)?)?;
    m.add_class::<PySafeOpen>()
}
